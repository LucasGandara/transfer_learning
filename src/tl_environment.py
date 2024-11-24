#!/usr/bin/env python3
# Authors: Lucas G. #

import math

import numpy as np
import rospy
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion

from src.respawn_goal import RespawnGoal


class Env(object):
    def __init__(self):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.position = Pose()

        self.state_size = 28
        self.action_size = 5

        self.init_goal = True  # First time the Goal is initialized

        # Node publisher
        self.cmd_vel_publisher = rospy.Publisher("cmd_vel", Twist, queue_size=5)

        # Node subscriptions
        rospy.Subscriber("odom", Odometry, self.odom_callback)
        self.reset_proxy = rospy.ServiceProxy("gazebo/reset_simulation", Empty)
        self.respawn_goal = RespawnGoal()

    def odom_callback(self, odom: Odometry):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(
            self.goal_y - self.position.y, self.goal_x - self.position.x
        )

        heading = goal_angle - yaw
        if heading > math.pi:
            heading -= 2 * math.pi

        elif heading < -math.pi:
            heading += 2 * math.pi

        self.heading = round(heading, 2)

    def get_goal_distance(self):
        goal_distance = round(
            math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2
        )

        return goal_distance

    def get_reward(self, state, done, action):
        yaw_reward = []
        obstacle_min_range = state[-2]
        current_distance = state[-3]
        heading = state[-4]

        for i in range(5):
            angle = -math.pi / 4 + heading + (math.pi / 8 * i) + math.pi / 2
            tr = 1 - 4 * math.fabs(
                0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0]
            )
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.goal_distance)

        if obstacle_min_range < 0.2:
            ob_reward = -5
        else:
            ob_reward = 0

        reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate) + ob_reward

        if done:
            rospy.loginfo("Collision!!")
            reward = -500
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal!!!! +1000 reward!!")
            reward = 1000
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.get_goal_distance()
            self.get_goalbox = False

        return reward

    def get_state(self, scan):
        scan_range = []
        heading = self.heading
        min_range = 0.13  # m: -> 13cm
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float("Inf"):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)
        if min_range > min(scan_range) > 0:
            done = True

        current_distance = self.get_goal_distance()
        if current_distance < 0.2:
            self.get_goalbox = True

        return (
            scan_range
            + [heading, current_distance, obstacle_min_range, obstacle_angle],
            done,
        )

    def step(self, action):
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1) / 2 - action) * max_angular_vel * 0.5

        cmd_vel = Twist()
        cmd_vel.linear.x = 0.15
        cmd_vel.angular.z = ang_vel
        self.pub_cmd_vel.publish(cmd_vel)

        data = None

        while data is None:
            try:
                data = rospy.wait_for_message("scan", LaserScan, timeout=5)
            except:
                pass

        state, done = self.get_state(data)
        reward = self.set_reward(state, done, action)

        return np.asarray(state), reward, done

    def reset(self):
        rospy.wait_for_service("/gazebo/reset_simulation", timeout=2)
        try:
            self.reset_proxy()
        except rospy.ServiceException as error:
            rospy.logerr(error)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message("scan", LaserScan, timeout=5)
            except:
                pass

        if self.init_goal:
            self.goal_x, self.goal_y = self.respawn_goal.get_position()
            self.init_goal = False

        self.goal_distance = self.get_goal_distance()
        state, done = self.get_state(data)


if __name__ == "__main__":
    rospy.init_node("tl_environment", anonymous=True)
    env = Env()
    env.reset()
