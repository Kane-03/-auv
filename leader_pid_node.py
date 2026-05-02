#!/usr/bin/env python3
"""
领航者 PID 节点 (Robot 1 - Leader)
参数已按 PID_recover.py 与 bluerov2h_origin_gazebo.xacro 物理约束对齐:
  - force_limits : [30, 25, 15]  N / Nm
  - kp/ki/kd     : [8/0.5/2, 5/0.5/1.5, 3/0.2/4]
  - max_rpm      : 100  (Kr=0.0012 → 单推进器约 12N)
  - dt           : 实测时间差，>0.2s 截断并重置积分
  - integral_limit: output_limits * 0.4  (抗饱和)
职责:
  1. 跟踪预设的正弦波参考轨迹
  2. 将自身实时位姿发布到 /formation/leader_pose，供跟随者使用
"""
import rospy
import numpy as np
import tf.transformations as tf
from sensor_msgs.msg import Imu
from gazebo_msgs.msg import LinkStates
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from geometry_msgs.msg import PoseStamped


class PID_Controller:
    """带限幅和抗饱和功能的 PID 控制器（与 PID_recover.py 完全一致）"""

    def __init__(self, kp, ki, kd, output_limits=None):
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.output_limits = (np.array(output_limits)
                              if output_limits is not None
                              else np.array([30.0, 25.0, 15.0]))
        self.integral   = np.zeros(3)
        self.last_error = np.zeros(3)

    def reset(self):
        self.integral   = np.zeros(3)
        self.last_error = np.zeros(3)

    def update(self, error, dt):
        if dt <= 0:
            return np.zeros(3)
        self.integral += error * dt
        # 积分项贡献上限为总输出的 40%
        integral_limit = self.output_limits * 0.4
        self.integral  = np.clip(self.integral, -integral_limit, integral_limit)
        derivative      = (error - self.last_error) / dt
        self.last_error = error.copy()
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(output, -self.output_limits, self.output_limits)


class LeaderPIDNode:
    def __init__(self):
        rospy.init_node('leader_pid_node')
        self.robot_id = rospy.get_param('~robot_id', 'bluerov2h_origin1')

        # 轨迹参数
        self.spiral_center_x = rospy.get_param('~traj_center_x', 0.0)
        self.spiral_center_y = rospy.get_param('~traj_center_y',  -1.0)
        self.start_time      = rospy.get_time()

        # PID（来自 PID_recover.py）
        self.pid = PID_Controller(
            kp=[8.0,  5.0,  3.0],
            ki=[0.5,  0.5,  0.2],
            kd=[2.0,  1.5,  4.0],
            output_limits=[30.0, 25.0, 15.0]
        )

        # 推力分配矩阵（来自 PID_recover.py，与 xacro 推进器坐标一致）
        self.B = np.array([
            [ 0.6691,  0.6691, -0.6691, -0.6691],
            [ 0.7431, -0.7431,  0.7431, -0.7431],
            [ 0.1732, -0.1732, -0.1651,  0.1651]
        ])
        self.A = np.linalg.pinv(self.B)

        # max_rpm=100 对应单推进器推力约 12N（来自 PID_recover.py）
        self.max_rpm = 100

        # 状态缓存
        self.current_position   = [0.0, 0.0, 0.0]
        self.current_euler      = [0.0, 0.0, 0.0]
        self.latest_state_valid = False
        self.imu_link_index     = None

        # 实测 dt 计时（来自 PID_recover.py）
        self.last_time = rospy.get_time()

        self._init_ros()

    def _init_ros(self):
        rospy.Subscriber(f'/{self.robot_id}/imu', Imu,self.imu_callback, queue_size=10)
        rospy.Subscriber('/gazebo/link_states', LinkStates,self.link_states_callback, queue_size=50)
        self.thruster_pubs = [rospy.Publisher(f'/{self.robot_id}/thrusters/{i}/input',FloatStamped, queue_size=10)for i in range(8)]
        self.leader_pose_pub = rospy.Publisher('/formation/leader_pose', PoseStamped, queue_size=10)
        
        rospy.loginfo(f'[Leader] robot_id={self.robot_id}')

    def _trajectory_generator(self, t):
        """
        生成参考轨迹 [x, y, yaw]
        y_dot 修正为 a*cos(a*t+π/2)，与 sin(a*t+π/2) 的导数一致（来自 PID_recover.py）
        """
        a       = 0.03
        ref_x   = self.spiral_center_x + a * t
        ref_y   = self.spiral_center_y + np.sin(a * t + np.pi / 2)
        x_dot   = a
        y_dot   = a * np.cos(a * t + np.pi / 2)
        ref_yaw = np.arctan2(y_dot, x_dot)
        return np.array([ref_x, ref_y, ref_yaw])

    def imu_callback(self, msg):
        q     = [msg.orientation.x, msg.orientation.y,
                 msg.orientation.z, msg.orientation.w]
        euler = tf.euler_from_quaternion(q, 'szyx')
        # 'szyx' 返回 [yaw, pitch, roll] → 映射到 [roll, pitch, yaw]
        self.current_euler      = [euler[2], euler[1], euler[0]]
        self.latest_state_valid = True

    def link_states_callback(self, msg):
        if self.imu_link_index is None:
            try:
                self.imu_link_index = msg.name.index(
                    f'{self.robot_id}::{self.robot_id}/imu_link')
            except ValueError:
                return
        idx = self.imu_link_index
        self.current_position = [
            msg.pose[idx].position.x,
            msg.pose[idx].position.y,
            msg.pose[idx].position.z
        ]
        self.latest_state_valid = True

    def _publish_leader_pose(self):
        msg = PoseStamped()
        msg.header.stamp       = rospy.Time.now()
        msg.header.frame_id    = 'world'
        msg.pose.position.x    = self.current_position[0]
        msg.pose.position.y    = self.current_position[1]
        msg.pose.position.z    = self.current_position[2]
        msg.pose.orientation.z = self.current_euler[2]   # yaw 存入 z
        self.leader_pose_pub.publish(msg)

    def control_step(self):
        if not self.latest_state_valid:
            return

        current_time = rospy.get_time()

        # 实测 dt（来自 PID_recover.py）
        dt = current_time - self.last_time
        if dt <= 0.001:
            return
        if dt > 0.2:
            dt = 0.1
            self.pid.reset()   # 异常大 dt 时重置积分防止突变
        self.last_time = current_time

        t = current_time - self.start_time

        ref_state  = self._trajectory_generator(t)
        curr_state = np.array([self.current_position[0],
                               self.current_position[1],
                               self.current_euler[2]])

        # 世界系误差 -> 体坐标系（来自 PID_recover.py）
        error_world    = ref_state - curr_state
        error_world[2] = (error_world[2] + np.pi) % (2 * np.pi) - np.pi

        psi = self.current_euler[2]
        R_world_to_body = np.array([
            [ np.cos(psi),  np.sin(psi), 0.0],
            [-np.sin(psi),  np.cos(psi), 0.0],
            [ 0.0,          0.0,         1.0]
        ])
        error_body = R_world_to_body @ error_world

        u = self.pid.update(error_body, dt).reshape(3, 1)

        self._publish_thrusters(u)
        self._publish_leader_pose()

        if int(t * 10) % 20 == 0:
            rospy.loginfo(
                f'[Leader t={t:.1f}s] dt={dt:.3f}s | '
                f'Ref: x={ref_state[0]:.2f} y={ref_state[1]:.2f} '
                f'yaw={np.degrees(ref_state[2]):.1f}° | '
                f'Curr: x={curr_state[0]:.2f} y={curr_state[1]:.2f} '
                f'yaw={np.degrees(curr_state[2]):.1f}° | '
                f'F(body): Fx={u[0,0]:.2f}N Fy={u[1,0]:.2f}N Mz={u[2,0]:.2f}Nm'
            )

    def _publish_thrusters(self, forces):
        def thrust_to_rpm(arr):
            Kr = 0.0012
            return np.where(np.abs(arr) < 1e-6,
                            0.0,
                            np.sign(arr) * np.sqrt(np.abs(arr) / Kr))
        u1_to_u4   = thrust_to_rpm(self.A @ forces)
        u5_to_u8   = np.zeros((4, 1))
        all_thrust = np.vstack((u1_to_u4, u5_to_u8)).flatten()
        for i, val in enumerate(all_thrust):
            msg      = FloatStamped()
            msg.data = float(np.clip(val, -self.max_rpm, self.max_rpm))
            self.thruster_pubs[i].publish(msg)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.control_step()
            rate.sleep()


if __name__ == '__main__':
    node = LeaderPIDNode()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
