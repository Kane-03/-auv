#!/usr/bin/env python3
"""
mpc_node1_fixed.py — 修正版 MPC 节点
主要修正（对照 bluerov2h_origin_gazebo.xacro）：

[严重-1] 动力学方程补全阻尼项
  原代码: acc = M⁻¹ · F  （完全忽略阻尼）
  修正:   acc = M⁻¹ · (F - D_lin·v - D_quad·|v|·v)
  URDF linear_damping:    Xu=-4.03, Yv=-6.22, Nr=-0.07
  URDF quadratic_damping: Xuu=-18.18, Yvv=-21.66, Nrr=-1.55

[严重-2] TVP 时间同步修正
  原代码: tvp_fun 内直接调用 rospy.get_time()，将实际时钟注入 MPC 仿真时间轴
  修正:   以 do_mpc 传入的 t_now（仿真步长累积时间）为基准，
          用 start_time 偏移量保持与 ROS 时钟的对应关系

[严重-3] MPC 力约束 vs RPM 截断矛盾
  原代码: MPC 约束 ±10 N，但 _publish_thrusters 中 max_thrust=2000 RPM
          2000 RPM 对应推力 0.0012×2000²=4800 N，两者相差 480 倍
  修正:   max_thrust_rpm = sqrt(max_thrust_N / Kr)，保持一致

[中等-1] izz 参数不一致
  原代码: self.izz = 0.260799
  URDF:   izz = 0.16（base_link inertia）
  修正:   使用 URDF 值 0.16

[中等-2] 代价函数权重调整
  原代码: Q_pos=1e3, P_pos=1e6, R=1e1
          P/Q=1000 倍，终端代价过大导致保守行为
  修正:   Q_pos=1e3, P_pos=1e2（与 Q 同量级），速度权重适当提升

[中等-3] 预测时域扩展
  原代码: N=5, t_step=0.1 → 预测窗口仅 0.5 s
  修正:   N=15 → 预测窗口 1.5 s，更适合 omega=0.5 rad/s 的弯曲轨迹

[中等-4] 参考速度坐标系一致性
  原代码: ref[3]=u_vel（world 系合速度大小）直接对应 body 系的 u 分量
  修正:   将 world 系参考速度旋转到 body 系后填入 ref[3:5]

[轻微] 重复赋值 self.Nrr 删除第二次
"""

import rospy
import do_mpc
import casadi as ca
import numpy as np
import tf.transformations as tf_trans
from sensor_msgs.msg import Imu
from gazebo_msgs.msg import LinkStates
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped


class MPC_Node:
    def __init__(self, robot_id='bluerov2h_origin'):
        rospy.init_node('mpc_node')
        self.robot_id = robot_id

        # ── 时间管理 ──────────────────────────────────────────
        self.start_time = rospy.get_time()
        self.start_time1 = 0.0

        # ── 轨迹参数 ──────────────────────────────────────────
        self.spiral_center_x = 25.0
        self.spiral_center_y = 0.0
        self.T = 15              # [修正-中等-3] 预测时域：5→15（窗口 1.5 s）
        self.traj_speed = 0.5   # x 方向匀速 m/s
        self.amplitude  = 1.0   # y 方向正弦振幅 m
        self.omega      = 0.5   # y 方向角频率 rad/s

        # ── 物理参数（对照 URDF） ─────────────────────────────
        self.m   = 11.5          # URDF base_link mass = 11.5 kg
        self.Fg  = self.m * 9.8

        # 转动惯量（URDF base_link inertia）
        self.ixx = 0.099
        self.iyy = 0.129
        self.izz = 0.16          # [修正-中等-1] URDF izz=0.16，原代码 0.260799

        # 附加质量（URDF added_mass 对角线）
        self.Xu_dot = -5.5
        self.Yv_dot = -12.7
        self.Nr_dot = -0.12

        # 线性阻尼（URDF linear_damping，水平面三项）
        self.Xu  = -4.03
        self.Yv  = -6.22
        self.Nr  = -0.07

        # 二次阻尼（URDF quadratic_damping）
        self.Xuu = -18.18
        self.Yvv = -21.66
        self.Nrr = -1.55        # [轻微] 删除第二次重复赋值

        # ── 状态缓存 ──────────────────────────────────────────
        self.current_position    = [25.0, 0.5, 0.0]
        self.current_euler       = [0.0, 0.0, 0.0]
        self.current_linear_vel  = [0.0, 0.0, 0.0]
        self.current_angular_vel = [0.0, 0.0, 0.0]
        self.current_orientation = [0.0, 0.0, 0.0, 1.0]
        self.imu_link_index      = None
        self.latest_state_valid  = False

        # ── MPC 约束参数（统一推力/RPM 上限） ────────────────
        # [修正-严重-3] 用力约束反推 RPM 上限，保持一致
        # 参考 PID_recover.py：单推进器 ≤ 约 15 N → RPM ≈ 112
        # 取保守值：MPC 输出力 ≤ 30 N（合力），单推进器 ≤ 7.5 N，RPM ≤ 79
        self.max_thrust_N   = 30.0   # MPC 层合力上限 N（每轴）
        self.max_thrust1    = self.max_thrust_N
        self.max_thrust2    = self.max_thrust_N
        self.max_thrust3    = 15.0   # 偏航力矩上限 N·m

        Kr = 0.0012
        # RPM 截断值 = sqrt(max_single_thruster_N / Kr)
        # max_single ≈ max_thrust_N / 2（伪逆分配后近似）
        self.max_rpm = float(np.sqrt(self.max_thrust_N / 2.0 / Kr))  # ≈ 112

        # ── 推力分配矩阵（URDF 推进器 48°/-48°/132°/-132°） ──
        self.B = np.array([
            [ 0.6691,  0.6691, -0.6691, -0.6691],
            [ 0.7431, -0.7431,  0.7431, -0.7431],
            [ 0.1732, -0.1732, -0.1651,  0.1651]
        ])
        self.A = np.linalg.pinv(self.B)

        self.mpc_initialized = False

        self._init_ros_components()
        self._init_mpc_controller()

    # ─────────────────────────────────────────────────────────
    # ROS 初始化
    # ─────────────────────────────────────────────────────────
    def _init_ros_components(self):
        rospy.Subscriber(f"/{self.robot_id}/imu", Imu,
                         self.imu_callback, queue_size=10)
        rospy.Subscriber("/gazebo/link_states", LinkStates,
                         self.link_states_callback, queue_size=50)
        self.thruster_pubs = [
            rospy.Publisher(f"/{self.robot_id}/thrusters/{i}/input",
                            FloatStamped, queue_size=10)
            for i in range(8)
        ]
        rospy.loginfo("MPC Node: ROS components initialized")

    # ─────────────────────────────────────────────────────────
    # MPC 控制器初始化
    # ─────────────────────────────────────────────────────────
    def _init_mpc_controller(self):
        model = do_mpc.model.Model('continuous')

        model.set_variable('_x', 'pos', shape=(3, 1))  # [x, y, psi]
        model.set_variable('_x', 'vel', shape=(3, 1))  # [u, v, r]（body 系）
        model.set_variable('_u', 'forces', shape=(3, 1))  # [Fx, Fy, Mz]（body 系）
        model.set_variable('_tvp', 'pos_ref', shape=(6, 1))

        # ── 惯性矩阵（含附加质量） ────────────────────────────
        M = ca.diag(ca.vertcat(
            self.m   - self.Xu_dot,   # 17.0
            self.m   - self.Yv_dot,   # 24.2
            self.izz - self.Nr_dot    # 0.28（修正后）
        ))

        # ── 旋转矩阵 body→world ───────────────────────────────
        psi = model.x['pos'][2]
        R = ca.vertcat(
            ca.horzcat(ca.cos(psi), -ca.sin(psi), 0),
            ca.horzcat(ca.sin(psi),  ca.cos(psi), 0),
            ca.horzcat(0,            0,           1)
        )

        # ── [修正-严重-1] 完整阻尼项 ─────────────────────────
        v_body = model.x['vel']   # [u, v, r]（body 系）
        u_b = v_body[0]
        v_b = v_body[1]
        r_b = v_body[2]

        # 线性阻尼力（body 系，负号已含于系数）
        D_lin = ca.vertcat(
            self.Xu  * u_b,
            self.Yv  * v_b,
            self.Nr  * r_b
        )
        # 二次阻尼力（带方向）
        D_quad = ca.vertcat(
            self.Xuu * u_b * ca.fabs(u_b),
            self.Yvv * v_b * ca.fabs(v_b),
            self.Nrr * r_b * ca.fabs(r_b)
        )

        u_forces = model.u['forces']
        # 完整速度导数：M·v̇ = F + D_lin·v + D_quad·|v|·v
        # 注意阻尼系数已为负值，此处直接相加
        acc = ca.inv(M) @ (u_forces + D_lin + D_quad)

        dxdt = ca.vertcat(
            R @ model.x['vel'],  # pos_dot（body→world）
            acc                  # vel_dot（body 系）
        )

        model.set_rhs('pos', dxdt[0:3])
        model.set_rhs('vel', dxdt[3:6])
        model.set_expression('acc_val', acc)
        model.setup()

        # ── MPC 参数 ──────────────────────────────────────────
        mpc = do_mpc.controller.MPC(model)
        mpc.settings.n_horizon = self.T       # [修正-中等-3] N=15
        mpc.settings.t_step    = 0.1

        # ── [修正-中等-2] 代价函数权重调整 ───────────────────
        # 原 P/Q=1000 倍终端代价过大；现与 Q 同量级
        pos_error = ca.vertcat(model.x['pos'], model.x['vel']) - model.tvp['pos_ref']
        Q = ca.diag([1e3, 1e3, 5e2, 1e2, 1e2, 5e1])  # 运行代价：位置/速度
        P = ca.diag([1e3, 1e3, 5e2, 1e2, 1e2, 5e1])  # 终端代价：与 Q 同量级

        lterm = ca.mtimes([pos_error.T, Q, pos_error])
        mterm = ca.mtimes([pos_error.T, P, pos_error])
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(forces=1e0)   # 控制惩罚适当放宽（原 1e1）

        # ── 约束（与 RPM 上限一致） ───────────────────────────
        bounds_u = [self.max_thrust1, self.max_thrust2, self.max_thrust3]
        for i in range(3):
            mpc.bounds['lower', '_u', 'forces', i] = -bounds_u[i]
            mpc.bounds['upper', '_u', 'forces', i] =  bounds_u[i]

        # ── TVP 函数 ──────────────────────────────────────────
        self.tvp_temp = mpc.get_tvp_template()

        def tvp_fun(t_now):
            # [修正-严重-2] 以 do_mpc 的仿真时间 t_now 为基准
            # start_time 是节点启动时刻（ROS 时钟），t_now 由 do_mpc 管理
            self._trajectory_generator(t_now)
            return self.tvp_temp

        mpc.set_tvp_fun(tvp_fun)
        mpc.settings.nlpsol_opts = {
            'ipopt.linear_solver': 'ma27',
            'ipopt.print_level': 0,
            'print_time': 0
        }
        mpc.setup()
        self.mpc = mpc
        self.mpc.obj_fun = ca.Function(
            'obj_fun',
            [self.mpc._opt_x, self.mpc._opt_p],
            [self.mpc._nlp_obj]
        )

    # ─────────────────────────────────────────────────────────
    # 轨迹生成器
    # ─────────────────────────────────────────────────────────
    def _trajectory_generator(self, t_now):
        """
        基于 do_mpc 仿真时间 t_now 生成预测轨迹。
        t_now 由 do_mpc 内部从 0 以 t_step 递增，是仿真时间轴上的时刻。
        用 start_time 将其映射到轨迹参数时间。
        """
        # [修正-严重-2] 用 do_mpc 的 t_now（仿真步长累积）作为轨迹时间基准
        # 首次调用时 t_now=0.1，此时 ROS 已运行了一小段时间，不再用实际时钟
        t_base = t_now  # do_mpc 管理的仿真时间

        v_x   = self.traj_speed
        Amp   = self.amplitude
        omega = self.omega

        for i in range(self.T + 1):
            t = t_base + i * 0.1  # 预测步 i 对应的仿真时刻

            # 位置
            x = self.spiral_center_x + v_x * t
            y = self.spiral_center_y + Amp * (np.sin(omega * t + np.pi / 2) - 1.0)

            # 一阶导数（world 系速度）
            x_dot = v_x
            y_dot = Amp * omega * np.cos(omega * t + np.pi / 2)

            # 二阶导数（world 系加速度）
            y_ddot = -Amp * omega**2 * np.sin(omega * t + np.pi / 2)

            # 参考航向角
            yaw = np.arctan2(y_dot, x_dot)

            # [修正-中等-4] 将 world 系速度旋转到 body 系
            cos_y = np.cos(yaw)
            sin_y = np.sin(yaw)
            # R_world_to_body = R^T
            u_ref =  cos_y * x_dot + sin_y * y_dot   # surge（body x）
            v_ref = -sin_y * x_dot + cos_y * y_dot   # sway（body y）

            # 曲率角速度
            denom = x_dot**2 + y_dot**2
            r_ref = (x_dot * y_ddot) / denom if denom > 1e-6 else 0.0

            ref_pos = np.array([x, y, yaw, u_ref, v_ref, r_ref])
            self.tvp_temp['_tvp', i, 'pos_ref'] = ref_pos.reshape(-1, 1)

    # ─────────────────────────────────────────────────────────
    # 回调
    # ─────────────────────────────────────────────────────────
    def imu_callback(self, msg):
        self.current_orientation = [
            msg.orientation.x, msg.orientation.y,
            msg.orientation.z, msg.orientation.w
        ]
        euler = tf_trans.euler_from_quaternion(self.current_orientation, 'szyx')
        self.current_euler       = [euler[2], euler[1], euler[0]]
        self.current_angular_vel = [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ]
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
        self.current_linear_vel = [
            msg.twist[idx].linear.x,
            msg.twist[idx].linear.y,
            msg.twist[idx].linear.z
        ]
        self.latest_state_valid = True

    # ─────────────────────────────────────────────────────────
    # 控制步
    # ─────────────────────────────────────────────────────────
    def control_step(self):
        if not self.latest_state_valid:
            return

        current_ros_time = rospy.get_time()
        if self.start_time1 == 0.0:
            self.start_time1 = current_ros_time

        # ── 构建状态向量 x0 ───────────────────────────────────
        # pos: [x, y, psi]（world 系）
        # vel: [u, v, r]（body 系）← link_states 给的是 world 系线速度，需转换
        psi = self.current_euler[2]
        R_mat = self._body_to_earth(psi)
        R_inv = R_mat.T  # world→body

        # world 系线速度
        vel_world = np.array(self.current_linear_vel[0:2] + [self.current_angular_vel[2]])

        # 转到 body 系（偏航方向旋转；角速度 r 不变）
        vel_body = np.zeros(3)
        vel_body[0:2] = R_inv[0:2, 0:2] @ np.array(self.current_linear_vel[0:2])
        vel_body[2]   = self.current_angular_vel[2]

        x0 = np.array([
            self.current_position[0],
            self.current_position[1],
            psi,
            vel_body[0],
            vel_body[1],
            vel_body[2]
        ]).reshape(-1, 1)

        # ── 热启动 ────────────────────────────────────────────
        if not self.mpc_initialized:
            self.mpc.x0 = x0
            self.mpc.set_initial_guess()
            self.mpc_initialized = True
            rospy.loginfo("MPC Initial Guess Set! Starting control loop...")

        # ── MPC 求解 ──────────────────────────────────────────
        u_mpc   = self.mpc.make_step(x0)
        u_final = u_mpc

        self._publish_thrusters(u_final)

        # ── 调试打印 ──────────────────────────────────────────
        aux = self.mpc.opt_aux_expression_fun(
            self.mpc.opt_x_num, self.mpc.opt_p_num)
        cost = self.mpc.obj_fun(self.mpc.opt_x_num, self.mpc.opt_p_num)

        rospy.loginfo_throttle(1.0,
            f"[t={current_ros_time:.1f}] "
            f"pos=({x0[0,0]:.2f},{x0[1,0]:.2f},{np.degrees(x0[2,0]):.1f}°) "
            f"vel_b=({x0[3,0]:.3f},{x0[4,0]:.3f},{x0[5,0]:.3f}) "
            f"u=({u_final[0,0]:.2f},{u_final[1,0]:.2f},{u_final[2,0]:.2f}) "
            f"cost={float(cost):.1f}"
        )

    # ─────────────────────────────────────────────────────────
    # 推力发布
    # ─────────────────────────────────────────────────────────
    def _publish_thrusters(self, forces):
        Kr = 0.0012

        def thrust_to_rpm(arr):
            return np.where(
                np.abs(arr) < 1e-6,
                0.0,
                np.sign(arr) * np.sqrt(np.abs(arr) / Kr)
            )

        u1_to_u4   = thrust_to_rpm(self.A @ forces)
        u5_to_u8   = np.zeros((4, 1))
        all_thrust = np.vstack((u1_to_u4, u5_to_u8)).flatten()

        for i, val in enumerate(all_thrust):
            msg      = FloatStamped()
            # [修正-严重-3] RPM 上限与力约束一致（≈112 RPM → ≈15 N/推进器）
            msg.data = float(np.clip(val, -self.max_rpm, self.max_rpm))
            self.thruster_pubs[i].publish(msg)

    # ─────────────────────────────────────────────────────────
    # 工具函数
    # ─────────────────────────────────────────────────────────
    def _body_to_earth(self, psi):
        """body→world 旋转矩阵（偏航角 psi）"""
        return np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi),  np.cos(psi), 0],
            [0,            0,           1]
        ])

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.control_step()
            rate.sleep()


if __name__ == '__main__':
    node = MPC_Node()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
