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
        super().__init__("Navigator")

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
            (1.5, 1.5),
            (1.5, 0.5),
            (1.5, -0.5),
            (1.5, -1.5),
            (0.5, -1.5),
            (-0.5, -1.5),
            (-1.5, -1.5),
            (-1.5, -0.5),
            (-1.5, 0.5),
            (-1.5, 1.5),
            (-0.5, 1.5),
            (0.5, 1.5)
        ]
        # self.waypoints = [
        #     (1.5, -0.5),
        #     (1.5, -1.5),
        #     (0.5, -1.5),
        #     (-0.5, -1.5),
        #     (-1.5, -1.5),
        #     (-1.5, -0.5),
        #     (-1.5, 0.5),
        #     (-1.5, 1.5),
        #     (-0.5, 1.5),
        #     (0.5, 1.5),
        #     (1.5, 1.5),
        #     (1.5, 0.5)
        # ]

        self.current_waypoint = 0
        self.distance_safe = 0.5
        self.max_speed = 0.26
        self.max_angular_speed = 1.82
        self.goal_tolerance = 0.5
        self.bubble_radius_deg = 35
        self.get_logger().info(f"The '{self.get_name()}' node is initialised.")


    def lidar_callback(self, scan_data:LaserScan):
        self.scan_data = scan_data
    
    def odom_callback(self, msg_data:Odometry):
        pose = msg_data.pose.pose
        (roll, pitch, yaw) = quaternion_to_euler(pose.orientation) 
        self.current_x = pose.position.x 
        self.current_y = pose.position.y
        self.current_yaw = yaw

    def find_best_gap(self, goal_angle_local):
        if not self.scan_data:
            return 0.0
        
        ranges = np.array(self.scan_data.ranges)
        ranges_length = len(ranges)
        # replace inf, nan, 0.0 with maximum range
        ranges = np.where(np.isinf(ranges) | np.isnan(ranges) | (ranges <= 0.05), self.scan_data.range_max, ranges)

        # Only get the front 180 degrees
        angle_increment = self.scan_data.angle_increment
        rear_start = int((math.pi/2 - self.scan_data.angle_min)/angle_increment)
        rear_end = int((3*math.pi/2 - self.scan_data.angle_min)/angle_increment)

        ranges[rear_start:rear_end] = 1000.0 #self.scan_data.range_max

        min_idx = np.argmin(ranges)
        self.danger_distance = ranges[min_idx]

        ranges[rear_start:rear_end] = 0.0

        bubble_radius_idx = int(math.radians(self.bubble_radius_deg) / angle_increment)
        for i in range(-bubble_radius_idx, bubble_radius_idx+1):
            idx = int((min_idx + i) % ranges_length)
            if 0 <= idx < ranges_length:
                ranges[idx] = 0.0

        safe_indices = np.where(ranges > self.distance_safe)[0]

        if len(safe_indices) == 0:
            # TRAPPED
            spin_dir = 1.57 * np.sign(goal_angle_local)
            if spin_dir == 0.0:
                spin_dir = -1.0
            return 1.57 * spin_dir
        
        wrapped_goal_angle = goal_angle_local
        if wrapped_goal_angle < 0:
            wrapped_goal_angle += 2 * math.pi

        goal_idx = int(wrapped_goal_angle / angle_increment)
        goal_idx = max(0, min(ranges_length - 1, goal_idx))

        circular_diffs = np.minimum(np.abs(safe_indices - goal_idx), ranges_length - np.abs(safe_indices - goal_idx))
        best_idx = safe_indices[np.argmin(circular_diffs)]

        # Convert back to an angle
        best_angle = best_idx * angle_increment
        
        # Convert back to standard ROS Right-Hand Rule (-pi to pi) for the steering controller
        if best_angle > math.pi:
            best_angle -= 2 * math.pi
            
        return best_angle

    def navigation_control(self):
        if self.current_waypoint >= len(self.waypoints):
            self.get_logger().info('All zone visited!')
            self.on_shutdown()
            return
        print(f"Current waypoint goal {str(self.current_waypoint)}")

        goal_x, goal_y = self.waypoints[self.current_waypoint]
        distance_to_goal = math.hypot(goal_x - self.current_x, goal_y - self.current_y)

        # Check if the waypoint has been reached
        if distance_to_goal < self.goal_tolerance:
            self.get_logger().info(f"Reached waypoing {str(self.current_waypoint)}")
            self.current_waypoint += 1
            return

        # Calculate goal
        goal_angle_global = math.atan2(goal_y - self.current_y, goal_x - self.current_x)
        goal_angle_local = goal_angle_global - self.current_yaw
        goal_angle_local = math.atan2(math.sin(goal_angle_local), math.cos(goal_angle_local))

        # Get best angle to steer towards
        safe_steering_angle = self.find_best_gap(goal_angle_local)

        msg = TwistStamped()
        msg.twist.angular.z = 2.0 * safe_steering_angle
        msg.twist.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, msg.twist.angular.z))

        if hasattr(self, 'danger_distance'):
            
            if self.danger_distance < 0.15:
                # STAGE 1: THE PANIC GEAR
                # The wall is practically touching the bumper!
                # Back up straight to get some breathing room.
                self.get_logger().warn(f'REVERSING! Wall at {self.danger_distance:.2f}m')
                msg.twist.linear.x = -0.1
                msg.twist.angular.z = 0.0 
                
            elif self.danger_distance < 0.35:
                # STAGE 2: THE SPIN-IN-PLACE BRAKE
                # The wall is close. Kill ALL forward momentum and just spin the chassis.
                self.get_logger().warn(f'BRAKING! Wall at {self.danger_distance:.2f}m')
                msg.twist.linear.x = 0.05  
                
            else:
                # STAGE 3: NORMAL RACING
                if abs(safe_steering_angle) < 0.4: 
                    msg.twist.linear.x = self.max_speed
                elif abs(safe_steering_angle) < 0.8:
                    msg.twist.linear.x = self.max_speed * 0.6
                else:
                    msg.twist.linear.x = 0.1
        print(f"CURRENT SPEED: {str(msg.twist.linear.x)}, Current Angular Speed {str(msg.twist.angular.z)}")
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

if __name__ == "__main__":
    main()