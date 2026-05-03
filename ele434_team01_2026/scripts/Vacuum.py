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

        # self.waypoints = [
        #     (1.5, 1.5),
        #     (1.5, 0.5),
        #     (1.5, -0.5),
        #     (1.5, -1.5),
        #     (0.5, -1.5),
        #     (-0.5, -1.5),
        #     (-1.5, -1.5),
        #     (-1.5, -0.5),
        #     (-1.5, 0.5),
        #     (-1.5, 1.5),
        #     (-0.5, 1.5),
        #     (0.5, 1.5)
        # ]
        self.waypoints = [
            (1.25, 1.25),
            (1.25, 0.25),
            (1.25, -0.25),
            (1.25, -1.25),
            (0.25, -1.25),
            (-0.5, -1.25),
            (-1.25, -1.25),
            (-1.25, -0.25),
            (-1.25, 0.25),
            (-1.25, 1.25),
            (-0.25, 1.25),
            (0.25, 1.25)
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
        self.evade_direction = 0.0

        self.current_waypoint = 0
        self.distance_safe = 0.4
        self.max_speed = 0.26
        self.max_angular_speed = 1.82
        self.goal_tolerance = 0.2
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

    def navigation_control(self):
        if self.current_waypoint >= len(self.waypoints):
            self.get_logger().info('All zones visited! Track Complete.')
            self.on_shutdown()
            return

        # 1. Waypoint Tracking
        goal_x, goal_y = self.waypoints[self.current_waypoint]
        distance_to_goal = math.hypot(goal_x - self.current_x, goal_y - self.current_y)

        if distance_to_goal < self.goal_tolerance:
            self.get_logger().info(f"Reached Area {self.current_waypoint + 1}")
            self.current_waypoint += 1
            return

        # Calculate angle to goal
        goal_angle_global = math.atan2(goal_y - self.current_y, goal_x - self.current_x)
        goal_angle_local = goal_angle_global - self.current_yaw
        goal_angle_local = math.atan2(math.sin(goal_angle_local), math.cos(goal_angle_local))

        msg = TwistStamped()

        # 2. Safety Check for LiDAR
        if not self.scan_data:
            return

        # Clean LiDAR data
        ranges = np.array(self.scan_data.ranges)
        ranges = np.where(np.isinf(ranges) | np.isnan(ranges) | (ranges <= 0.05), 3.0, ranges)

        # 3. Create a matching array of Angles and normalize to -pi to pi 
        # (This mathematically cures the 0.0 degree wrap-around bug forever!)
        angles = np.linspace(self.scan_data.angle_min,
                             self.scan_data.angle_min + self.scan_data.angle_increment * (len(ranges) - 1),
                             len(ranges))
        angles = np.arctan2(np.sin(angles), np.cos(angles))

        # 4. Define the 3 Sectors (in radians)
        # Center: -35 to +35 degrees. Left: +35 to +90. Right: -35 to -90.
        center_mask = (angles > -0.6) & (angles < 0.6)  
        left_mask = (angles >= 0.6) & (angles < 1.57)   
        right_mask = (angles <= -0.6) & (angles > -1.57) 

        # 5. Find the closest object in each sector
        dist_center = np.min(ranges[center_mask]) if np.any(center_mask) else 3.0
        dist_left = np.min(ranges[left_mask]) if np.any(left_mask) else 3.0
        dist_right = np.min(ranges[right_mask]) if np.any(right_mask) else 3.0

        # --- THE NEW WHISKER CHECK ---
        # Find the absolute closest point in the entire front 180-degree view
        front_mask = center_mask | left_mask | right_mask
        front_ranges = np.where(front_mask, ranges, 3.0)
        min_dist_front = np.min(front_ranges)
        min_angle_front = angles[np.argmin(front_ranges)]
        # -----------------------------

        # 6. THE ROOMBA BRAIN (With Virtual Whiskers)
        if min_dist_front < 0.25:
            # WHISKER OVERRIDE: We are physically scraping a wall!
            # Ignore the waypoint and forcefully push away from the closest point.
            self.get_logger().warn(f'SCRAPING! Repelling from wall at {min_dist_front:.2f}m')
            msg.twist.linear.x = 0.05  
            # If the wall is on our left (positive angle), steer right. Else, steer left.
            msg.twist.angular.z = -self.max_angular_speed if min_angle_front > 0 else self.max_angular_speed

        elif dist_center > 0.4:
            # CLEAR PATH: The center is open and the sides are safe. Race to waypoint!
            self.evade_direction = 0.0
            msg.twist.linear.x = self.max_speed
            msg.twist.angular.z = 2.0 * goal_angle_local
            if dist_left < 0.32:
                self.get_logger().info('Passing object on Left...')
                msg.twist.angular.z = 0.3  # Push gently right
            elif dist_right < 0.32:
                self.get_logger().info('Passing object on Right...')
                msg.twist.angular.z = -0.3   # Push gently left
            else:
                # Both sides are safe. Race to the waypoint!
                msg.twist.angular.z = 2.0 * goal_angle_local
            
        else:
            # CENTER BLOCKED: The wall is in front of us, but not touching the bumper yet.
            self.get_logger().warn(f'EVADING! Center blocked at {dist_center:.2f}m')
            msg.twist.linear.x = 0.01

            if self.evade_direction == 0.0:
                self.evade_direction = 1.0 if dist_left > dist_right else -1.0
            
            if dist_left > dist_right:
                msg.twist.angular.z = self.evade_direction * self.max_angular_speed  # Hard Left
            else:
                msg.twist.angular.z = self.evade_direction * self.max_angular_speed # Hard Right

        # Clamp steering to the hardware limits of the TurtleBot
        msg.twist.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, msg.twist.angular.z))

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
        time.sleep(3)
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