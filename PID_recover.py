#!/usr/bin/env python3
import rospy
import numpy as np
import tf.transformations as tf
from sensor_msgs.msg import Imu
from gazebo_msgs.msg import LinkStates
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from std_msgs.msg import Float64MultiArray

# ============================================================
# 修正说明（对照 bluerov2h_origin_gazebo.xacro）：
#
# 1. force_limits: 5N -> [30, 25, 15]
#    URDF: rotor_constant Kr=0.0012, max_rpm 实测单推进器最大推力
#    F = Kr * rpm^2 => rpm=60 时 F≈4.3N，4 个推进器合力约 15-30N
#    原来 5N 极度保守，导致 Kp=50 的大增益被限幅完全抵消，控制无效
#
# 2. max_rpm: 2000 -> 63
#    URDF rotor_constant=0.0012，推力限幅 30N 等效
#    max_thrust_per_thruster ≈ 30/4 * sqrt(合力系数) → rpm = sqrt(F/Kr) ≈ 63
#    原来 max_rpm=2000 对应 F=4800N/推进器，完全脱离实际
#
# 3. dt: 硬编码 0.1 -> 实测时间差
#    硬编码导致积分累计误差；加入 dt > 0.2s 保护防止突变
#
# 4. PID 增益重新匹配新限幅：
#    原 kp=[50,15,10] 在 5N 限幅下实际无效（任何 >0.1m 误差即饱和）
#    新增益 kp=[8,5,3] 与 force_limits=[30,25,15] 配合使用
#    误差 1m → Fx=8N（不饱和），有合理的比例响应
#    kd 增大以补偿水下阻尼不足带来的振荡
#
# 5. 旋转矩阵方向：
#    world->body 转换应为 R.T（R 是 body->world 的旋转矩阵）
#    修正后 R_world_to_body 明确注释说明
# ============================================================

class PID_Controller:
    """带限幅和抗饱和功能的 PID 控制器"""

    def __init__(self, kp, ki, kd, output_limits=None):
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)

        # output_limits: [max_Fx(N), max_Fy(N), max_Mz(Nm)]
        self.output_limits = np.array(output_limits) if output_limits is not None else np.array([30.0, 25.0, 15.0])

        self.integral = np.zeros(3)
        self.last_error = np.zeros(3)

    def update(self, error, dt):
        if dt <= 0:
            return np.zeros(3)

        # 积分更新
        self.integral += error * dt

        # 积分限幅（抗饱和）：积分项贡献上限为总输出的 40%
        integral_limit = self.output_limits * 0.4
        self.integral = np.clip(self.integral, -integral_limit, integral_limit)

        # 微分项
        derivative = (error - self.last_error) / dt
        self.last_error = error.copy()

        # PID 输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # 输出限幅
        output = np.clip(output, -self.output_limits, self.output_limits)

        return output


class PID_Node:
    def __init__(self, robot_id='bluerov2h_origin'):
        rospy.init_node('pid_node')
        self.robot_id = robot_id

        # ---------- 轨迹参数 ----------
        self.spiral_center_x = 25.0
        self.spiral_center_y = 0.0
        self.start_time = rospy.get_time()

        # ---------- PID 参数与限幅配置 ----------
        # 依据 URDF 物理约束重新标定：
        #   rotor_constant Kr = 0.0012 (xacro)
        #   推力 F = Kr * rpm^2
        #   水平推进器合力（surge/sway）：4 推进器，矢量合成约 15–30 N
        #   偏航力矩：力臂 ≈ 0.14–0.17 m，最大约 30*0.17 ≈ 5 Nm，设定保守值 15 Nm
        force_limits = [30.0, 25.0, 15.0]  # [Surge Force(N), Sway Force(N), Yaw Torque(Nm)]

        self.pid = PID_Controller(
            kp=[8.0,  5.0,  3.0],   # 1m 误差产生 8/5/3 N，不饱和，有比例响应
            ki=[0.5,  0.5,  0.2],   # 小积分防止稳态超调
            kd=[2.0,  1.5,  4.0],   # 较强阻尼，抵抗水下振荡
            output_limits=force_limits
        )

        # ---------- 物理与分配矩阵 ----------
        # URDF: thruster 0 rpy(0,0,48°), thruster 1 rpy(0,0,-48°)
        #        thruster 2 rpy(0,0,132°), thruster 3 rpy(0,0,-132°)
        # cos(48°)=0.6691, sin(48°)=0.7431
        # cos(48°)=0.6691, sin(132°)=sin(48°)=0.7431
        # 力矩臂近似值（推进器位置 x/y 分量决定，见 xacro 坐标）：
        #   T0: (+0.1506, -0.0975) × force direction  → l_z ≈ +0.1732
        #   T1: (+0.1506, +0.0975)                    → l_z ≈ -0.1732
        #   T2: (-0.1343, -0.1013)                    → l_z ≈ -0.1651
        #   T3: (-0.1343, +0.1013)                    → l_z ≈ +0.1651
        # B 矩阵列为各推进器对 [Fx, Fy, Mz] 的贡献（体坐标系）
        self.B = np.array([
            [ 0.6691,  0.6691, -0.6691, -0.6691],  # Surge (cos48°)
            [ 0.7431, -0.7431,  0.7431, -0.7431],  # Sway  (sin48°，注意符号)
            [ 0.1732, -0.1732, -0.1651,  0.1651]   # Yaw torque（力矩臂×推力方向）
        ])
        self.A = np.linalg.pinv(self.B)  # 伪逆用于推力分配

        # ---------- 推进器 RPM 上限 ----------
        # URDF rotor_constant Kr=0.0012, F=Kr*rpm^2
        # force_limits 中最大推力 30N → 分配到单推进器约 F_single ≤ 30/2 ≈ 15N
        # rpm_max = sqrt(15 / 0.0012) ≈ 112，留余量设为 100
        # 注意：原代码 max_rpm=2000 对应 F=4800N，与 force_limits=5N 完全矛盾
        self.max_rpm = 100  # 对应单推进器最大推力约 12N

        # ---------- 状态缓存 ----------
        self.current_position = [0.0, 0.0, 0.0]
        self.current_euler = [0.0, 0.0, 0.0]
        self.latest_state_valid = False
        self.last_time = rospy.get_time()

        self._init_ros_components()

    def _init_ros_components(self):
        rospy.Subscriber(f"/{self.robot_id}/imu", Imu, self.imu_callback, queue_size=10)
        rospy.Subscriber("/gazebo/link_states", LinkStates, self.link_states_callback, queue_size=50)
        self.thruster_pubs = [
            rospy.Publisher(f"/{self.robot_id}/thrusters/{i}/input", FloatStamped, queue_size=10)
            for i in range(8)
        ]

    def _trajectory_generator(self, t):
        """螺旋线轨迹生成：返回 [ref_x, ref_y, ref_yaw]"""
        a = 0.03
        ref_x = self.spiral_center_x + a * t
        ref_y = self.spiral_center_y + np.sin(a * t + np.pi / 2)
        x_dot = a
        y_dot = a * np.cos(a * t + np.pi / 2)  # 修正：原为 -a*sin(a*t)，与 sin(a*t+π/2) 一致
        ref_yaw = np.arctan2(y_dot, x_dot)
        return np.array([ref_x, ref_y, ref_yaw])

    def imu_callback(self, msg):
        orientation = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        euler = tf.euler_from_quaternion(orientation, 'szyx')
        # 'szyx' 返回顺序：[yaw(z), pitch(y), roll(x)]，映射到 [roll, pitch, yaw]
        self.current_euler = [euler[2], euler[1], euler[0]]
        self.latest_state_valid = True

    def link_states_callback(self, msg):
        try:
            idx = msg.name.index(f'{self.robot_id}::{self.robot_id}/imu_link')
            self.current_position = [
                msg.pose[idx].position.x,
                msg.pose[idx].position.y,
                msg.pose[idx].position.z
            ]
            self.latest_state_valid = True
        except ValueError:
            pass

    def control_step(self):
        if not self.latest_state_valid:
            return

        current_time = rospy.get_time()

        # --- 修正 3: 使用实测 dt，防止首帧或卡帧时 dt 过大 ---
        dt = current_time - self.last_time
        if dt <= 0.001:
            return  # 防止除零
        if dt > 0.2:
            dt = 0.1  # 异常大 dt（如暂停后恢复）截断为标称值，同时重置积分
            self.pid.integral = np.zeros(3)

        t_from_start = current_time - self.start_time

        # 1. 参考状态
        ref_state = self._trajectory_generator(t_from_start)

        # 2. 当前状态 [x, y, yaw]
        curr_state = np.array([
            self.current_position[0],
            self.current_position[1],
            self.current_euler[2]
        ])

        # 3. 世界坐标系误差
        error_world = ref_state - curr_state
        # 偏航角误差归一化到 [-π, π]（在坐标转换前处理）
        error_world[2] = (error_world[2] + np.pi) % (2.0 * np.pi) - np.pi

        # 4. 转换到体坐标系（Body Frame）
        # R_body_to_world = [[cos(ψ), -sin(ψ), 0],
        #                    [sin(ψ),  cos(ψ), 0],
        #                    [0,       0,      1]]
        # world -> body: 取 R 的转置
        psi = self.current_euler[2]
        R_world_to_body = np.array([
            [ np.cos(psi), np.sin(psi), 0.0],
            [-np.sin(psi), np.cos(psi), 0.0],
            [ 0.0,         0.0,         1.0]
        ])
        error_body = R_world_to_body @ error_world

        # 5. PID 计算（内部已有限幅）
        u_final = self.pid.update(error_body, dt).reshape(3, 1)

        # 6. 发布推进器指令
        self._publish_thrusters(u_final)

        self.last_time = current_time

        # 调试打印（每 2s 一次）
        if int(t_from_start * 10) % 20 == 0:
            print(f"[t={t_from_start:.1f}s] dt={dt:.3f}s | "
                  f"Ref: x={ref_state[0]:.2f} y={ref_state[1]:.2f} yaw={np.degrees(ref_state[2]):.1f}° | "
                  f"Curr: x={curr_state[0]:.2f} y={curr_state[1]:.2f} yaw={np.degrees(curr_state[2]):.1f}° | "
                  f"Force(body): Fx={u_final[0,0]:.2f}N Fy={u_final[1,0]:.2f}N Mz={u_final[2,0]:.2f}Nm")

    def _publish_thrusters(self, forces):
        """
        力 -> RPM 转换并发布
        URDF: thrust = rotor_constant * rpm * |rpm|
              即 F = Kr * rpm^2（带符号方向），Kr=0.0012
        """
        def thrust_to_rpm(thrust_array):
            Kr = 0.0012  # 与 xacro rotor_constant 一致
            abs_thrust = np.abs(thrust_array)
            return np.where(
                abs_thrust < 1e-6,
                0.0,
                np.sign(thrust_array) * np.sqrt(abs_thrust / Kr)
            )

        # 水平推进器 0-3：通过 B 矩阵伪逆分配力
        u1_to_u4 = self.A @ forces          # shape (4,1)
        u1_to_u4 = thrust_to_rpm(u1_to_u4)

        # 垂直推进器 4-7：深度保持（此脚本不控制深度）
        u5_to_u8 = np.zeros((4, 1))

        all_thrusters = np.vstack((u1_to_u4, u5_to_u8)).flatten()

        for i, val in enumerate(all_thrusters):
            msg = FloatStamped()
            # max_rpm=100 对应单推进器推力约 12N，与 force_limits 匹配
            msg.data = float(np.clip(val, -self.max_rpm, self.max_rpm))
            self.thruster_pubs[i].publish(msg)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.control_step()
            rate.sleep()


if __name__ == '__main__':
    node = PID_Node()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
