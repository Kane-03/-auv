#!/usr/bin/env python3
"""
跟随者 PID 节点 (Robot 2 / Robot 3 - Follower)
参数已按 PID_recover.py 与 bluerov2h_origin_gazebo.xacro 物理约束对齐:
  - force_limits : [30, 25, 15]  N / Nm
  - kp/ki/kd     : [8/0.5/2, 5/0.5/1.5, 3/0.2/4]
  - max_rpm      : 100
  - dt           : 实测时间差，>0.2s 截断并重置积分
  - integral_limit: output_limits * 0.4

队形偏移（领航者体坐标系，旋转到世界系后使用）:
  Robot 2:  dx=-5m, dy=+2m  （领航者左后方）
  Robot 3:  dx=-5m, dy=-2m  （领航者右后方）
"""
import rospy
import numpy as np
import tf.transformations as tf
from sensor_msgs.msg import Imu
from gazebo_msgs.msg import LinkStates
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from geometry_msgs.msg import PoseStamped


class PID_Controller:
    """与 PID_recover.py 完全一致"""

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
        integral_limit = self.output_limits * 0.4
        self.integral  = np.clip(self.integral, -integral_limit, integral_limit)
        derivative      = (error - self.last_error) / dt
        self.last_error = error.copy()
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(output, -self.output_limits, self.output_limits)


class FollowerPIDNode:
    FORMATION_OFFSETS = {
        2: np.array([-2.0,  1.0]),   # Robot 2：左后方
        3: np.array([-2.0, -1.0]),   # Robot 3：右后方
    }

    def __init__(self):
        rospy.init_node('follower_pid_node')
        self.robot_id    = rospy.get_param('~robot_id',    'bluerov2h_origin2')
        self.robot_index = int(rospy.get_param('~robot_index', 2))

        if self.robot_index not in self.FORMATION_OFFSETS:
            rospy.logerr(f'[Follower] 未知 robot_index={self.robot_index}，支持 2 或 3')
            rospy.signal_shutdown('invalid robot_index')
            return

        self.formation_offset = self.FORMATION_OFFSETS[self.robot_index]
        rospy.loginfo(
            f'[Follower {self.robot_index}] robot_id={self.robot_id}, '
            f'队形偏移={self.formation_offset}'
        )

        # PID（来自 PID_recover.py）
        self.pid = PID_Controller(
            kp=[8.0,  5.0,  3.0],
            ki=[0.5,  0.5,  0.2],
            kd=[2.0,  1.5,  4.0],
            output_limits=[30.0, 25.0, 15.0]
        )

        # 推力分配矩阵（来自 PID_recover.py）
        self.B = np.array([
            [ 0.6691,  0.6691, -0.6691, -0.6691],
            [ 0.7431, -0.7431,  0.7431, -0.7431],
            [ 0.1732, -0.1732, -0.1651,  0.1651]
        ])
        self.A = np.linalg.pinv(self.B)

        # max_rpm=100（来自 PID_recover.py）
        self.max_rpm = 100

        # 状态缓存
        self.current_position   = [0.0, 0.0, 0.0]
        self.current_euler      = [0.0, 0.0, 0.0]
        self.latest_state_valid = False
        self.imu_link_index     = None

        # 领航者位姿缓存
        self.leader_pose     = None
        self.leader_received = False

        # 实测 dt 计时（来自 PID_recover.py）
        self.last_time = rospy.get_time()

        self._init_ros()

    def _init_ros(self):
        rospy.Subscriber(f'/{self.robot_id}/imu', Imu,
                         self.imu_callback, queue_size=10)
        rospy.Subscriber('/gazebo/link_states', LinkStates,
                         self.link_states_callback, queue_size=50)
        rospy.Subscriber('/formation/leader_pose', PoseStamped,
                         self.leader_pose_callback, queue_size=10)
        self.thruster_pubs = [
            rospy.Publisher(f'/{self.robot_id}/thrusters/{i}/input',
                            FloatStamped, queue_size=10)
            for i in range(8)
        ]

    def imu_callback(self, msg):
        q     = [msg.orientation.x, msg.orientation.y,
                 msg.orientation.z, msg.orientation.w]
        euler = tf.euler_from_quaternion(q, 'szyx')
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

    def leader_pose_callback(self, msg):        # 存储领航者位姿
        self.leader_pose     = [msg.pose.position.x,
                                msg.pose.position.y,
                                msg.pose.orientation.z]   # z 存 yaw
        self.leader_received = True

    def _compute_formation_target(self):
        """
        目标点 = 领航者位置 + R(leader_yaw) × 队形偏移
        偏移在领航者体坐标系中定义，旋转到世界系后使用，
        转弯时跟随者始终保持在"身后"而不漂移到侧面。
        """
        lx, ly, lyaw = self.leader_pose     # 领航者位姿（世界系）
        dx, dy = self.formation_offset      # 队形偏移（领航者体坐标系）
        cos_y  = np.cos(lyaw)               # 旋转矩阵元素（领航者朝向）
        sin_y  = np.sin(lyaw)               # 旋转矩阵元素（领航者朝向）
        world_dx = cos_y * dx - sin_y * dy  # 旋转偏移到世界系
        world_dy = sin_y * dx + cos_y * dy  # 旋转偏移到世界系
        return np.array([lx + world_dx, ly + world_dy, lyaw])   # 目标位姿（世界系）

    def control_step(self):
        if not self.latest_state_valid:
            return
        if not self.leader_received:
            rospy.logwarn_throttle(
                2.0, f'[Follower {self.robot_index}] 等待领航者位姿...')
            return

        current_time = rospy.get_time()

        # 实测 dt（来自 PID_recover.py）
        dt = current_time - self.last_time
        if dt <= 0.001:
            return
        if dt > 0.2:
            dt = 0.1
            self.pid.reset()
        self.last_time = current_time

        target     = self._compute_formation_target()
        curr_state = np.array([self.current_position[0],
                               self.current_position[1],
                               self.current_euler[2]])

        # 世界系误差 -> 体坐标系（来自 PID_recover.py）
        error_world    = target - curr_state
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

        t = current_time - self.last_time   # 仅用于打印节奏
        if int(current_time * 10) % 20 == 0:
            rospy.loginfo(
                f'[Follower {self.robot_index}] dt={dt:.3f}s | '
                f'Target: x={target[0]:.2f} y={target[1]:.2f} '
                f'yaw={np.degrees(target[2]):.1f}° | '
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
    node = FollowerPIDNode()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
