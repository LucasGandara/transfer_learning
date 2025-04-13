import math
import os
import sys
from math import pi

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion

from src.reward_functions import (
    angular_reward_function,
    combined_reward_function,
    linear_reward_function,
)


class DebugReward(object):
    def __init__(self):
        rospy.Subscriber("odom", Odometry, self.odom_callback, queue_size=10)
        self.orientation = -100
        self.goal_angle = -100
        self.heading = -100
        self.state = [0.0 for i in range(29)]  # Updated for time steps
        self.angular_action = 0.15
        self.steps = 0
        self.max_steps = 450  # From config default

        self.angular_reward = 0.0
        self.linear_reward = -10.0
        self.reward = -10.0

        self.goal_y = 0.0
        self.goal_x = 1.5
        self.goal_distance = 0.0
        self.start_goal_distance = 1.5

        self.orientation_publisher = rospy.Publisher(
            "/orientation", Float64, queue_size=10
        )

        self.goal_angle_publisher = rospy.Publisher(
            "/goal_angle", Float64, queue_size=10
        )
        self.heading_publisher = rospy.Publisher("/heading", Float64, queue_size=10)
        self.time_steps_publisher = rospy.Publisher(
            "/time_steps", Float64, queue_size=10
        )

        self.cmd_vel_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.angle_reward_publisher = rospy.Publisher(
            "/angle_reward", Float64, queue_size=10
        )
        self.linear_reward_publisher = rospy.Publisher(
            "/linear_reward", Float64, queue_size=10
        )
        self.reward_publisher = rospy.Publisher("/reward", Float64, queue_size=10)
        self.goal_distance_publisher = rospy.Publisher(
            "/goal_distance", Float64, queue_size=10
        )

        rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_callback)

    def cmd_vel_callback(self, cmd_vel: Twist):
        self.angular_action = cmd_vel.angular.z

    def get_goal_distance(self):
        self.current_goal_distance = round(
            math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2
        )
        rospy.loginfo("Current goal distance {}".format(self.current_goal_distance))
        return self.current_goal_distance

    def odom_callback(self, odom: Odometry):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        self.orientation = yaw
        self.goal_angle = math.atan2(
            self.goal_y - self.position.y, self.goal_x - self.position.x
        )

        self.heading = self.goal_angle - yaw
        if self.heading > math.pi:
            self.heading -= 2 * math.pi

        elif self.heading < -math.pi:
            self.heading += 2 * math.pi

        self.heading = round(self.heading, 2)

    def get_state(self, scan):
        scan_range = []
        heading = self.heading

        min_range = 0.135  # m: -> 13cm
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float("Inf") or scan.ranges[i] == float("inf"):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]) or scan.ranges[i] == float("nan"):
                scan_range.append(0.0)
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:
            done = True

        self.scan_range = scan_range
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float("Inf") or scan.ranges[i] == float("inf"):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]) or scan.ranges[i] == float("nan"):
                scan_range.append(0.0)
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:
            done = True

        current_distance = self.get_goal_distance()
        self.steps += 1
        normalized_time = self.steps / self.max_steps

        self.state = scan_range + [heading, current_distance, normalized_time]

    def get_reward(self):
        current_distance = self.state[-1]
        theta = abs(self.state[-2])
        omega = self.angular_action
        alpha = 1

        self.angular_reward = angular_reward_function(theta, omega, alpha)
        self.linear_reward = linear_reward_function(
            self.start_goal_distance, current_distance
        )
        self.reward = combined_reward_function(
            theta,
            omega,
            self.start_goal_distance,
            current_distance,
            w_angular=1.0,
            w_distance=3.0,
            alpha=1.0,
        )

    def pub_msgs(self):
        orientation_msg = Float64()
        orientation_msg.data = self.orientation
        goal_angle_msg = Float64()
        goal_angle_msg.data = self.goal_angle
        heading_msg = Float64()
        heading_msg.data = self.heading
        time_steps_msg = Float64()
        time_steps_msg.data = self.steps / self.max_steps

        angular_reward_msg = Float64()
        angular_reward_msg.data = self.angular_reward

        linear_reward_msg = Float64()
        linear_reward_msg.data = self.linear_reward

        reward_msg = Float64()
        reward_msg.data = self.reward

        goal_distance_msg = Float64()
        goal_angle_msg.data = self.current_goal_distance

        self.orientation_publisher.publish(orientation_msg)
        self.goal_angle_publisher.publish(goal_angle_msg)
        self.heading_publisher.publish(heading_msg)
        self.time_steps_publisher.publish(time_steps_msg)
        self.angle_reward_publisher.publish(angular_reward_msg)
        self.linear_reward_publisher.publish(linear_reward_msg)
        self.reward_publisher.publish(reward_msg)
        self.goal_distance_publisher.publish(goal_distance_msg)


if __name__ == "__main__":
    rospy.init_node("debug_reward", anonymous=True)
    rospy.loginfo("Debug reward node started")

    node = DebugReward()
    while True:
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message("scan", LaserScan, timeout=5)
            except:
                pass
        node.get_state(data)
        node.get_reward()
        node.pub_msgs()
        rospy.sleep(0.2)
