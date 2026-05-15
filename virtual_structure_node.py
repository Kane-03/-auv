#!/usr/bin/env python3
"""
虚拟结构中心轨迹生成器 (Virtual Structure Center Generator)
职责:
  1. 生成虚拟结构中心（Virtual Center）的参考轨迹
  2. 将虚拟结构中心的实时位姿发布到 /formation/virtual_center_pose，供所有跟随者使用
"""
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped

class VirtualStructureNode:
    def __init__(self):
        rospy.init_node('virtual_structure_node')
        
        # 轨迹参数 (可配置)
        self.spiral_center_x = rospy.get_param('~traj_center_x', 0.0)
        self.spiral_center_y = rospy.get_param('~traj_center_y', 0.0)
        self.traj_speed = rospy.get_param('~traj_speed', 0.5)   # 前进速度 m/s
        self.amplitude = rospy.get_param('~amplitude', 1.0)     # 波形振幅
        self.omega = rospy.get_param('~omega', 0.5)             # 角频率 rad/s

        self.start_time = rospy.get_time()
        self.center_pose_pub = rospy.Publisher('/formation/virtual_center_pose', PoseStamped, queue_size=10)
        
        rospy.loginfo('[Virtual Structure] 虚拟结构中心轨迹生成器已启动.')

    def _trajectory_generator(self, t):
        """
        生成虚拟结构中心的参考轨迹 [x, y, yaw]
        y_dot = A * omega * cos(omega*t+π/2)，与 A*sin(omega*t+π/2) 的导数一致
        """
        v_x     = self.traj_speed   # 线速度 (x 方向恒定前进)
        A       = self.amplitude    # 振幅 (控制波形的大小)
        w       = self.omega    # 角频率 (控制波形的快慢)
        
        ref_x   = self.spiral_center_x + v_x * t    # x 方向匀速前进
        ref_y   = self.spiral_center_y + A * (np.sin(w * t + np.pi / 2) - 1.0)  # y 方向正弦波动，减去 1.0 保证初始 y 受参数约束而在 (0,0) 开始
        x_dot   = v_x   # x 方向速度恒定
        y_dot   = A * w * np.cos(w * t + np.pi / 2) # y 方向速度为正弦波的导数
        ref_yaw = np.arctan2(y_dot, x_dot)  # 计算航向角，使其始终朝向运动方向
        return np.array([ref_x, ref_y, ref_yaw])   

    def _publish_center_pose(self, state):
        msg = PoseStamped()
        msg.header.stamp       = rospy.Time.now()
        msg.header.frame_id    = 'world'
        msg.pose.position.x    = state[0]
        msg.pose.position.y    = state[1]
        msg.pose.position.z    = 0.0
        msg.pose.orientation.z = state[2]   # yaw 存入 z
        self.center_pose_pub.publish(msg)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            # 必须等待时钟启动
            if current_time > 0:
                # 只在刚启动时获取一下 start_time
                if not hasattr(self, "actual_start_time"):
                    self.actual_start_time = current_time
                    
                t = current_time - self.actual_start_time
                ref_state = self._trajectory_generator(t)
                self._publish_center_pose(ref_state)
                
            rate.sleep()

if __name__ == '__main__':
    node = VirtualStructureNode()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
