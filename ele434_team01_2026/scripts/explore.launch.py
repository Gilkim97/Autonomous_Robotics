#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions 

from sensor_msgs.msg import LaserScan 
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import time
from part2_navigation_modules.tb3_tools import quaternion_to_euler 
from math import sqrt, pow, pi 
import math
import numpy as np 

class AutonomousNavigation(Node):

    def __init__(self):
        super.__init__("Navigator")

        self.lidar_sub = self.create_subscription(
            msg_type = LaserScan,
            topic = "/scan",
            callback = self.lidar_callback,
            qos_profile = 10,
        )
        self.vel_pub = self.create_publisher(
            msg_type = TwistStamped,
            topic = "/cmd_vel",
            qos_profile = 10,
        )
        self.odom_data = self.create_subscription(
            msg_type = Odometry,
            topic = "/odom",
            callback = self.odom_callback,
            qos_profile = 10,
        )

        self.timer = self.create_timer(
            timer_period_sec = 1/20,
            callback = self.navigation_control
        )

        # Set current state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.scan_data = None

        self.waypoints = [
            (1.5, -0.5),
            (1.5, -1.5),
            (0.5, -1.5),
            (-0.5, -1.5),
            (-1.5, -1.5),
            (-1.5, -0.5),
            (-1.5, 0.5),
            (-1.5, 1.5),
            (-0.5, 1.5),
            (0.5, 1.5),
            (1.5, 1.5),
            (1.5, 0.5)
        ]
        self.current_waypoint = 0

        self.k_goal = 1.0
        self.k_obstacle = 1.0
        self.distance_safe = 0.2
        self.max_speed = 0.26
        self.max_angular_speed = 1.8
        self.goal_tolerance = 0.3
        self.get_logger().info(f"The '{self.get_name()}' node is initialised.")


    def lidar_callback(self, scan_data:LaserScan):
        self.scan_data = scan_data
    
    def odom_callback(self, msg_data:Odometry):
        pose = msg_data.pose.pose
        (roll, pitch, yaw) = quaternion_to_euler(pose.orientation) 
        self.current_x = pose.position.x 
        self.current_y = pose.position.y
        self.current_yaw = abs(yaw)

    def force_obstacles(self):
        if not self.scan_data:
            return np.array([0.0, 0.0])
        
        force_x = 0.0
        force_y = 0.0
        
        ranges = self.scan_data.ranges
        angle_min = self.scan_data.angle_min
        angle_increment = self.scan_data.angle_increment

        for i, distance in enumerate(ranges):

            if distance < self.scan_data.range_min or distance < self.scan_data.range_max:
                continue

            if distance < self.distance_safe:
                angle = self.current_yaw + angle_increment*i + angle_min
                magnitude = self.k_obstacle*(1.0/distance**2.0)
                force_x -= magnitude * math.cos(angle)
                force_y -= magnitude * math.sin(angle)

        return np.array([force_x, force_y])
    
    def force_goal(self, goal_x, goal_y):
        error_x = goal_x - self.current_x
        error_y = goal_y - self.current_y

        distance = math.hypot(error_x, error_y)
        if distance == 0.0:
            return np.array([0.0, 0.0])
        
        force_x = self.k_goal * (error_x/distance)
        force_y = self.k_goal * (error_y/distance)

        return np.array([force_x, force_y])

    def navigation_control(self):
        if self.current_waypoint >= len(self.waypoints):
            self.get_logger().info('All zone visited!')
            self.on_shutdown()
            return
        
        goal_x, goal_y = self.waypoints[self.current_waypoint]

        distance_to_goal = math.hypot(goal_x - self.current_x, goal_y - self.current_y)
        if distance_to_goal < self.goal_tolerance:
            self.current_waypoint += 1
            return
        
        F_obstacle = self.force_obstacles()
        F_goal = self.force_goal(goal_x, goal_y)

        F_total = F_obstacle + F_goal

        desired_yaw = math.atan2(F_total[1], F_total[0]) - self.current_yaw
        yaw_error = math.atan2(math.sin(desired_yaw), math.cos(desired_yaw))
        
        msg = TwistStamped()
        msg.twist.angular.z = 1.5 * yaw_error

        if abs(yaw_error) < 0.5:
            speed = math.hypot(F_total[0], F_total[1])
            msg.twist.linear.x = min(speed, self.max_speed)
        else:
            msg.twist.linear.x = 0.0

        self.vel_pub.publish(msg)

    def on_shutdown(self):
        self.get_logger().info(
            "Stopping the robot..."
        )
        self.vel_pub.publish(TwistStamped()) 
        self.shutdown = True


def main(args=None):
    rclpy.init(
        args = args,
        signal_handler_options = SignalHandlerOptions.NO
    )
    node = AutonomousNavigation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutdown request detected..")
    finally:
        node.on_shutdown()
        while not node.shutdown():
            continue
        node.destroy_node()
        rclpy.shutdown()