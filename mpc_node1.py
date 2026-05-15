#!/usr/bin/env python3
import rospy
import do_mpc
import casadi as ca
import numpy as np
import tf.transformations as tf
from sensor_msgs.msg import Imu
from gazebo_msgs.msg import LinkStates
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R22

class MPC_Node:
    def __init__(self, robot_id='bluerov2h_origin'):
        rospy.init_node('mpc_node')
        self.robot_id = robot_id
        
        # ------------------ 时间与状态管理 ------------------
        self.current_time = 0
        self.start_time = rospy.get_time()
        self.start_time1 = 0.0 
        
        # 螺旋轨迹参数
        self.spiral_center_x = 25 
        self.spiral_center_y = 0 
        self.T = 5 # 预测时域
        self.traj_speed = 0.5
        self.amplitude = 1.0
        self.omega = 0.5

        # 物理参数 
        self.m = 11.543
        self.Fg = self.m * 9.8
        self.ixx = 0.199
        self.iyy = 0.229799
        self.izz = 0.260799
        
        self.Xu_dot = -5.5
        self.Yv_dot = -12.7
        self.Nr_dot = -0.12
        self.Xu = -4.03
        self.Yv = -6.22
        self.Nr = -0.07
        self.Xuu = -18.18
        self.Yvv = -21.66 
        self.Nrr = -1.55
        self.Nrr = -1.55

        # 状态缓存
        self.current_position = [25, 0.5, 0]
        self.current_euler = [0, 0, 0]
        self.current_linear_vel = [0, 0, 0]
        self.current_angular_vel = [0, 0, 0]
        self.current_orientation = [0, 0, 0, 1]
        self.linear_acceleration = [0, 0, 0]
        self.imu_link_index = None
        self.latest_state_valid = False
        self.gravity_vector_world = np.array([0, 0, 9.8])

        
        # ------------------ MPC配置 ------------------
        self.max_thrust = 2000
        self.max_thrust1 = 10
        self.max_thrust2 = 10
        self.max_thrust3 = 10
        
        # 推力分配矩阵
        self.B = np.array([
            [0.6691, 0.6691, -0.6691, -0.6691],
            [0.7431, -0.7431, 0.7431, -0.7431],
            [0.1732, -0.1732, -0.1651, 0.1651]
        ])
        self.A = np.linalg.pinv(self.B)
        # 热启动标志位
        self.mpc_initialized = False

        # ------------------ ROS 组件 ------------------
        self._init_ros_components()
        self._init_mpc_controller()

    def _init_ros_components(self):
        rospy.Subscriber(f"/{self.robot_id}/imu", Imu, self.imu_callback, queue_size=10)
        rospy.Subscriber("/gazebo/link_states", LinkStates, self.link_states_callback, queue_size=50)
                
        # 推进器发布器
        self.thruster_pubs = [
            rospy.Publisher(f"/{self.robot_id}/thrusters/{i}/input", FloatStamped, queue_size=10)
            for i in range(8)
        ]
        rospy.loginfo("MPC Node: ROS components initialized")

    def _init_mpc_controller(self):
        model = do_mpc.model.Model('continuous')
        
        model.set_variable('_x', 'pos', shape=(3,1)) 
        model.set_variable('_x', 'vel', shape=(3,1)) 
        model.set_variable('_u', 'forces', shape=(3,1)) 
        
        # TVP
        model.set_variable('_tvp', 'pos_ref', shape=(6,1))

        # 动力学方程构建
        M = ca.diag(ca.vertcat(
            self.m - self.Xu_dot,
            self.m - self.Yv_dot,
            self.izz - self.Nr_dot
        ))
        
        u_vel, v_vel = model.x['vel'][0], model.x['vel'][1]
        
        # 旋转矩阵
        psi = model.x['pos'][2]
        R = ca.vertcat(
            ca.horzcat(ca.cos(psi), -ca.sin(psi), 0),
            ca.horzcat(ca.sin(psi),  ca.cos(psi), 0),
            ca.horzcat(0,            0,           1)
        )
        R_inv = ca.inv(R)
        
        # 旋转矩阵导数相关项
        r = model.x['vel'][2]
        inv_R_dot = ca.vertcat(
            ca.horzcat(0, -r, 0),
            ca.horzcat(r,  0, 0),
            ca.horzcat(0,  0, 0)
        )

        u_forces = model.u['forces']
        #加速度（可根据公式填写或者自行设计，只保留了M_inv乘力，可用上面的一些定义）
        acceleration = ca.inv(M) @ u_forces  

        dxdt = ca.vertcat(
            R @ model.x['vel'],  # pos_dot
            acceleration         # vel_dot
        )

        model.set_rhs('pos', dxdt[0:3])
        model.set_rhs('vel', dxdt[3:6])
        model.set_expression('acc_val', ca.inv(M) @ u_forces) # 查看dompc内部数据
        model.setup()

        # MPC 设置
        mpc = do_mpc.controller.MPC(model)
        mpc.settings.n_horizon = self.T
        mpc.settings.t_step = 0.1
        
        # 代价函数
        pos_error = ca.vertcat(model.x['pos'], model.x['vel']) - model.tvp['pos_ref']
        P = ca.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
        Q = ca.diag([1e3, 1e3, 1e3, 1e1, 1e1, 1e1])
        
        lterm = (ca.mtimes([pos_error.T, Q, pos_error]))
        mterm = (ca.mtimes([pos_error.T, P, pos_error]))
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(forces = 1e1)

        # 约束
        for i in range(3):
            mpc.bounds['lower', '_u', 'forces', i] = -[self.max_thrust1, self.max_thrust2, self.max_thrust3][i]
            mpc.bounds['upper', '_u', 'forces', i] =  [self.max_thrust1, self.max_thrust2, self.max_thrust3][i]

        # TVP 函数
        self.tvp_temp = mpc.get_tvp_template()
        def tvp_fun(t_now):
            self._trajectory_generator(t_now)
            return self.tvp_temp
        
        mpc.set_tvp_fun(tvp_fun)
        mpc.settings.nlpsol_opts = {'ipopt.linear_solver': 'ma27', 'ipopt.print_level': 0, 'print_time': 0}
        mpc.setup()
        self.mpc = mpc
        # 在MPC配置完成后创建目标函数计算器
        self.mpc.obj_fun = ca.Function('obj_fun', [self.mpc._opt_x, self.mpc._opt_p], [self.mpc._nlp_obj])

    def _trajectory_generator(self, t_now):
        """生成轨迹"""
        # 原有的时间重置逻辑
        if t_now == 0.1:
            self.start_time = rospy.get_time() - 0.1
        
        t_now_ros = rospy.get_time()
        
        for i in range(self.T + 1):
            t = t_now_ros + i * 0.1 - self.start_time
            
            # 轨迹参数
            v_x = self.traj_speed       # x方向匀速前进线速度 (m/s)
            Amp = self.amplitude        # y方向正弦波动振幅 (m)
            omega = self.omega          # y方向正弦波动角频率 (rad/s)
            
            # 位置计算 (正弦波轨迹)
            x = self.spiral_center_x + v_x * t                                  # 参考点x坐标
            y = self.spiral_center_y + Amp * (np.sin(omega * t + np.pi/2) - 1.0) # 参考点y坐标
            
            # 一阶导数计算 (即绝对坐标系下的速度)
            x_dot = v_x                                             # x方向速度
            y_dot = Amp * omega * np.cos(omega * t + np.pi/2)       # y方向速度
            
            # 二阶导数计算 (即绝对坐标系下的加速度)
            x_ddot = 0                                              # x方向加速度
            y_ddot = -Amp * omega * omega * np.sin(omega * t + np.pi/2)   # y方向加速度
            
            # 姿态与本体速度计算
            yaw = np.arctan2(y_dot, x_dot)    # 参考航向角 (沿轨迹切线方向)
            u_vel = np.sqrt(x_dot**2 + y_dot**2)    # 参考线速度 (合速度)
            r = (x_dot * y_ddot - y_dot * x_ddot)/(x_dot**2 + y_dot**2) if (x_dot**2 + y_dot**2) > 1e-6 else 0 # 参考偏航角速度 (曲率带来的角速度)
            
            # 组装完整的参考状态维度: [x, y, yaw, u, v, r] (v设为0假定期望无横滑移)
            ref_pos = np.array([x, y, yaw, u_vel, 0, r])
            self.tvp_temp['_tvp', i, 'pos_ref'] = ref_pos.reshape(-1,1)
            

    def imu_callback(self, msg):
        self.current_orientation = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        euler = tf.euler_from_quaternion(self.current_orientation, 'szyx')
        self.current_euler = [euler[2], euler[1], euler[0]] 
        self.current_angular_vel = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        self.linear_acceleration = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        self.latest_state_valid = True

    def link_states_callback(self, msg):
        if self.imu_link_index is None:
            try:
                self.imu_link_index = msg.name.index(f'{self.robot_id}::{self.robot_id}/imu_link')
            except ValueError:
                return
        self.current_position = [
            msg.pose[self.imu_link_index].position.x,
            msg.pose[self.imu_link_index].position.y,
            msg.pose[self.imu_link_index].position.z
        ]
        self.current_linear_vel = [
            msg.twist[self.imu_link_index].linear.x,
            msg.twist[self.imu_link_index].linear.y,
            msg.twist[self.imu_link_index].linear.z
        ]
        self.latest_state_valid = True


    def control_step(self):

        if not self.latest_state_valid:
            return

        current_ros_time = rospy.get_time()
        if self.start_time1 == 0.0:
            self.start_time1 = current_ros_time

        # 1. 构建状态向量 x0
        x0 = np.concatenate([
            np.array(self.current_position[0:2]),
            [self.current_euler[2]],
            np.array(self.current_linear_vel[0:2]),
            [self.current_angular_vel[2]],
        ]).reshape(-1,1)
        R = self._body_to_earth(self.current_euler)
        R_inv = R.T
        x0[3:6] = np.dot(R_inv, x0[3:6])
        # 第一次运行时的“热启动”逻辑
        if not self.mpc_initialized:
            # 1. 告诉 MPC 当前在哪里
            self.mpc.x0 = x0          
            # 2. 告诉 MPC 生成初始猜测 (这会根据系统方程预演未来的轨迹)
            self.mpc.set_initial_guess() 
            # 3. 标记为已初始化，以后不再运行这段
            self.mpc_initialized = True 
            rospy.loginfo("MPC Initial Guess Set! Starting control loop...")
        # 2. MPC 求解
        u_mpc = self.mpc.make_step(x0)
        u_final = u_mpc 
        # --- 新增：显示 acceleration 计算数值 ---
        aux_data = self.mpc.opt_aux_expression_fun(self.mpc.opt_x_num, self.mpc.opt_p_num)
        #（打印dompc内部数据用）
        print("-" * 30)
        print(f"Time: {current_ros_time:.2f}")
        print(f"Calculated Acceleration (Body Frame):")
        print(f"  x: {float(aux_data[1]):.4f} ")
        print(f"  y: {float(aux_data[2]):.4f} ")
        print(f"  z: {float(aux_data[3]):.4f} ")  
        # ---------------------------------------

        # u_final = np.array([0, 0, 0]).reshape(3, 1)
        # 4. 发布推进器指令
        print(u_final.flatten())
        self._publish_thrusters(u_final)
        
        # 打印信息
        # print(f"Time: {current_ros_time:.2f}, Pos: {x0[:2].flatten()}, Disturbance used: {self.latest_disturbance}")
        print("轨迹",self.tvp_temp['_tvp', 0])
        print("x,y,r :", x0[:3].flatten().tolist())  # 输出格式: [x, y, r]
        print("vel :", x0[3:6].flatten().tolist())  # 输出格式: [u, v, r]

        current_cost = self.mpc.obj_fun(self.mpc.opt_x_num, self.mpc.opt_p_num)
        print(f"Accumulated predicted cost: {float(current_cost):.2f}")

    def _publish_thrusters(self, forces):
        def thrust_to_rpm(thrust_array):
            Kr = 0.0012 
            abs_thrust = np.abs(thrust_array)
            return np.where(abs_thrust < 1e-6, 0.0, np.sign(thrust_array) * np.sqrt(abs_thrust / Kr))

        u1_to_u4 = self.A @ forces
        u1_to_u4 = thrust_to_rpm(u1_to_u4)
        u5_to_u8 = np.zeros((4, 1))
        all_thrusters = np.vstack((u1_to_u4, u5_to_u8)).flatten()
        
        for i, force in enumerate(all_thrusters):
            msg = FloatStamped()
            msg.data = np.clip(force, -self.max_thrust, self.max_thrust)
            self.thruster_pubs[i].publish(msg)

    def run(self):
        rate = rospy.Rate(10) # 10Hz
        while not rospy.is_shutdown():
            self.control_step()
            rate.sleep()   

    def _body_to_earth(self, euler_angles):
            """将体坐标系速度转换到大地坐标系"""
            psi = euler_angles[2]
            R = np.array([
                [np.cos(psi), -np.sin(psi), 0],
                [np.sin(psi),  np.cos(psi), 0],
                [0,            0,           1]
            ])

            return R

if __name__ == '__main__':
    node = MPC_Node()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass