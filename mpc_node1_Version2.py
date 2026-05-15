#!/usr/bin/env python3
"""MPC controller node for BlueROV2 Heavy (planar 3-DOF: x, y, yaw).

Updated to align actuator limits and thruster mapping with PID_recover.py and
with bluerov2h_origin_gazebo.xacro.

Key updates:
- Use URDF-consistent mass/inertia for planar model (m=11.5, izz=0.16).
- Add linear + quadratic damping terms (from xacro) into the dynamics.
- Align MPC input constraints with PID: [Fx, Fy, Mz] limits = [30, 25, 15].
- Replace unrealistic thruster command clipping (2000) with max_rpm=100.
- Keep same thruster allocation B (4 horizontal thrusters) and Kr=0.0012.
- Make body/world velocity handling explicit and consistent.

NOTE:
- The 15 Nm yaw limit matches PID_recover.py, but may exceed what max_rpm=100
  can generate. Thruster-level saturation will clip commands.
"""

import rospy
import do_mpc
import casadi as ca
import numpy as np
import tf.transformations as tf
from sensor_msgs.msg import Imu
from gazebo_msgs.msg import LinkStates
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped


class MPC_Node:
    def __init__(self, robot_id: str = 'bluerov2h_origin'):
        rospy.init_node('mpc_node')
        self.robot_id = robot_id

        # ------------------ Time / state ------------------
        self.start_time = rospy.get_time()
        self.imu_link_index = None
        self.latest_state_valid = False
        self.mpc_initialized = False

        # ------------------ Reference trajectory parameters ------------------
        self.center_x = 25.0
        self.center_y = 0.0
        self.traj_speed = 0.5
        self.amplitude = 1.0
        self.omega = 0.5

        # ------------------ Model parameters (from xacro) ------------------
        # URDF inertial (base_link): mass=11.5, izz=0.16.
        self.m = 11.5
        self.izz = 0.16

        # Added mass (xacro added_mass diag terms): 5.5, 12.7, yaw 0.12
        # Use convention: M = diag(m - Xu_dot, m - Yv_dot, izz - Nr_dot),
        # with Xu_dot, Yv_dot, Nr_dot negative.
        self.Xu_dot = -5.5
        self.Yv_dot = -12.7
        self.Nr_dot = -0.12

        # Damping (xacro)
        self.Xu = -4.03
        self.Yv = -6.22
        self.Nr = -0.07
        self.Xuu = -18.18
        self.Yvv = -21.66
        self.Nrr = -1.55

        # ------------------ MPC settings ------------------
        self.dt = 0.1
        self.n_horizon = 50  # 5s at 0.1s

        # Input limits aligned with PID_recover.py
        self.force_limits = np.array([30.0, 25.0, 15.0], dtype=float)  # [Fx, Fy, Mz]

        # Thruster model/limits aligned with PID_recover.py
        self.Kr = 0.0012
        self.max_rpm = 100.0

        # Thruster allocation (horizontal 0-3) aligned with PID_recover.py
        self.B = np.array(
            [
                [0.6691, 0.6691, -0.6691, -0.6691],
                [0.7431, -0.7431, 0.7431, -0.7431],
                [0.1732, -0.1732, -0.1651, 0.1651],
            ],
            dtype=float,
        )
        self.A = np.linalg.pinv(self.B)

        # ------------------ State cache ------------------
        self.current_position = np.zeros(3)              # world
        self.current_euler = np.zeros(3)                 # [roll,pitch,yaw]
        self.current_linear_vel_world = np.zeros(3)      # world
        self.current_angular_vel = np.zeros(3)           # body (from IMU)

        # ------------------ ROS wiring ------------------
        self._init_ros_components()
        self._init_mpc_controller()

    def _init_ros_components(self):
        rospy.Subscriber(f"/{self.robot_id}/imu", Imu, self.imu_callback, queue_size=10)
        rospy.Subscriber("/gazebo/link_states", LinkStates, self.link_states_callback, queue_size=50)

        self.thruster_pubs = [
            rospy.Publisher(f"/{self.robot_id}/thrusters/{i}/input", FloatStamped, queue_size=10)
            for i in range(8)
        ]
        rospy.loginfo("MPC Node: ROS components initialized")

    # ------------------ MPC setup ------------------
    def _init_mpc_controller(self):
        model = do_mpc.model.Model('continuous')

        # States: pos=[x,y,psi] (world), vel=[u,v,r] (body)
        model.set_variable('_x', 'pos', shape=(3, 1))
        model.set_variable('_x', 'vel', shape=(3, 1))

        # Control: [Fx,Fy,Mz] in body frame
        model.set_variable('_u', 'forces', shape=(3, 1))

        # Reference: [x_ref,y_ref,psi_ref,u_ref,v_ref,r_ref]
        model.set_variable('_tvp', 'ref', shape=(6, 1))

        # Mass matrix (planar)
        M = ca.diag(
            ca.vertcat(
                self.m - self.Xu_dot,
                self.m - self.Yv_dot,
                self.izz - self.Nr_dot,
            )
        )

        # Kinematics
        psi = model.x['pos'][2]
        R_b2w = ca.vertcat(
            ca.horzcat(ca.cos(psi), -ca.sin(psi), 0),
            ca.horzcat(ca.sin(psi),  ca.cos(psi), 0),
            ca.horzcat(0,            0,           1),
        )

        u = model.x['vel'][0]
        v = model.x['vel'][1]
        r = model.x['vel'][2]

        # Damping: tau_damp = - (D_lin*v + D_quad*|v|*v)
        Xu = abs(self.Xu)
        Yv = abs(self.Yv)
        Nr = abs(self.Nr)
        Xuu = abs(self.Xuu)
        Yvv = abs(self.Yvv)
        Nrr = abs(self.Nrr)

        tau_damp = ca.vertcat(
            -(Xu * u + Xuu * ca.fabs(u) * u),
            -(Yv * v + Yvv * ca.fabs(v) * v),
            -(Nr * r + Nrr * ca.fabs(r) * r),
        )

        tau = model.u['forces'] + tau_damp
        v_dot = ca.inv(M) @ tau

        x_dot = ca.vertcat(
            R_b2w @ model.x['vel'],  # pos_dot (world)
            v_dot,                   # vel_dot (body)
        )

        model.set_rhs('pos', x_dot[0:3])
        model.set_rhs('vel', x_dot[3:6])

        model.set_expression('tau_damp', tau_damp)
        model.set_expression('acc_val', v_dot)
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.settings.n_horizon = int(self.n_horizon)
        mpc.settings.t_step = float(self.dt)

        err = ca.vertcat(model.x['pos'], model.x['vel']) - model.tvp['ref']

        # Moderate weights to avoid over-aggressive control with saturation
        Q = ca.diag([2e3, 2e3, 1e3, 5e1, 5e1, 2e2])
        P = ca.diag([5e3, 5e3, 2e3, 1e2, 1e2, 4e2])
        mpc.set_objective(mterm=ca.mtimes([err.T, P, err]),
                          lterm=ca.mtimes([err.T, Q, err]))
        mpc.set_rterm(forces=1e-2)

        # Constraints aligned with PID
        for i in range(3):
            lim = float(self.force_limits[i])
            mpc.bounds['lower', '_u', 'forces', i] = -lim
            mpc.bounds['upper', '_u', 'forces', i] = lim

        # TVP
        self.tvp_temp = mpc.get_tvp_template()

        def tvp_fun(t_now):
            self._fill_reference()
            return self.tvp_temp

        mpc.set_tvp_fun(tvp_fun)
        mpc.settings.nlpsol_opts = {
            'ipopt.linear_solver': 'ma27',
            'ipopt.print_level': 0,
            'print_time': 0,
        }
        mpc.setup()
        self.mpc = mpc

        self.mpc.obj_fun = ca.Function('obj_fun',
                                       [self.mpc._opt_x, self.mpc._opt_p],
                                       [self.mpc._nlp_obj])

    # ------------------ Reference generation ------------------
    def _fill_reference(self):
        t0 = rospy.get_time() - self.start_time
        for k in range(int(self.n_horizon) + 1):
            t = t0 + k * self.dt

            v_x = self.traj_speed
            Amp = self.amplitude
            omega = self.omega

            x = self.center_x + v_x * t
            y = self.center_y + Amp * (np.sin(omega * t + np.pi / 2.0) - 1.0)

            x_dot = v_x
            y_dot = Amp * omega * np.cos(omega * t + np.pi / 2.0)
            x_ddot = 0.0
            y_ddot = -Amp * omega * omega * np.sin(omega * t + np.pi / 2.0)

            psi_ref = np.arctan2(y_dot, x_dot)

            speed = np.sqrt(x_dot ** 2 + y_dot ** 2)
            u_ref = speed
            v_ref = 0.0

            denom = (x_dot ** 2 + y_dot ** 2)
            r_ref = (x_dot * y_ddot - y_dot * x_ddot) / denom if denom > 1e-6 else 0.0

            ref = np.array([x, y, psi_ref, u_ref, v_ref, r_ref], dtype=float).reshape(-1, 1)
            self.tvp_temp['_tvp', k, 'ref'] = ref

    # ------------------ ROS callbacks ------------------
    def imu_callback(self, msg: Imu):
        q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        euler = tf.euler_from_quaternion(q, 'szyx')
        # Keep same mapping convention as original file
        self.current_euler = np.array([euler[2], euler[1], euler[0]], dtype=float)
        self.current_angular_vel = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=float
        )
        self.latest_state_valid = True

    def link_states_callback(self, msg: LinkStates):
        if self.imu_link_index is None:
            try:
                self.imu_link_index = msg.name.index(f'{self.robot_id}::{self.robot_id}/imu_link')
            except ValueError:
                return

        self.current_position = np.array(
            [
                msg.pose[self.imu_link_index].position.x,
                msg.pose[self.imu_link_index].position.y,
                msg.pose[self.imu_link_index].position.z,
            ],
            dtype=float,
        )
        self.current_linear_vel_world = np.array(
            [
                msg.twist[self.imu_link_index].linear.x,
                msg.twist[self.imu_link_index].linear.y,
                msg.twist[self.imu_link_index].linear.z,
            ],
            dtype=float,
        )
        self.latest_state_valid = True

    # ------------------ Control loop ------------------
    def control_step(self):
        if not self.latest_state_valid:
            return

        psi = float(self.current_euler[2])

        # world -> body for linear velocity
        R_world_to_body = np.array(
            [
                [np.cos(psi),  np.sin(psi), 0.0],
                [-np.sin(psi), np.cos(psi), 0.0],
                [0.0,          0.0,         1.0],
            ],
            dtype=float,
        )
        vel_body = R_world_to_body @ self.current_linear_vel_world.reshape(3, 1)

        x0 = np.array(
            [
                self.current_position[0],
                self.current_position[1],
                psi,
                float(vel_body[0]),
                float(vel_body[1]),
                float(self.current_angular_vel[2]),
            ],
            dtype=float,
        ).reshape(-1, 1)

        if not self.mpc_initialized:
            self.mpc.x0 = x0
            self.mpc.set_initial_guess()
            self.mpc_initialized = True
            rospy.loginfo("MPC initialized, entering control loop")

        u_mpc = self.mpc.make_step(x0)
        self._publish_thrusters(u_mpc)

    # ------------------ Actuation ------------------
    def _thrust_to_rpm(self, thrust: np.ndarray) -> np.ndarray:
        abs_thrust = np.abs(thrust)
        rpm = np.where(abs_thrust < 1e-9, 0.0, np.sign(thrust) * np.sqrt(abs_thrust / self.Kr))
        return rpm

    def _publish_thrusters(self, forces: np.ndarray):
        # Allocate to horizontal thrusters 0-3
        u1_to_u4_thrust = self.A @ forces
        u1_to_u4_rpm = self._thrust_to_rpm(u1_to_u4_thrust)

        # Vertical thrusters 4-7 not used
        u5_to_u8 = np.zeros((4, 1), dtype=float)

        all_cmd = np.vstack((u1_to_u4_rpm, u5_to_u8)).flatten()

        for i, cmd in enumerate(all_cmd):
            msg = FloatStamped()
            msg.data = float(np.clip(cmd, -self.max_rpm, self.max_rpm))
            self.thruster_pubs[i].publish(msg)

    def run(self):
        rate = rospy.Rate(int(1.0 / self.dt))
        while not rospy.is_shutdown():
            self.control_step()
            rate.sleep()


if __name__ == '__main__':
    node = MPC_Node()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass