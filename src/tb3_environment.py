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

from src.consts import get_stage, get_stage_name
from src.respawn_goal import RespawnGoal


class Env(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.position = Pose()

        self.steps = 0
        self.timeout = False

        self.state_size = 28
        self.action_size = 2

        self.get_goalbox = False
        self.init_goal = True  # First time the Goal is initialized

        self.past_distance = 0.0
        self.past_action = [0] * self.action_size

        # Node publisher
        self.cmd_vel_publisher = rospy.Publisher("cmd_vel", Twist, queue_size=5)

        # Node subscriptions
        rospy.Subscriber("odom", Odometry, self.odom_callback)
        self.reset_proxy = rospy.ServiceProxy("gazebo/reset_simulation", Empty)

        stage = get_stage(cfg["stage"])
        rospy.loginfo("Environment stage: {}".format(get_stage_name(cfg["stage"])))
        self.respawn_goal = RespawnGoal(stage)

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

        self.heading = round(heading, 3)

    def get_goal_distance(self):
        goal_distance = round(
            math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2
        )
        self.past_distance = goal_distance

        return goal_distance

    def get_reward(self, state, done):
        current_distance = state[-1]
        heading = abs(state[-2])

        distance_rate = self.past_distance - current_distance

        if distance_rate > 0:
            reward = 200.0 * distance_rate

        if distance_rate <= 0:
            reward = -8.0

        self.past_distance = current_distance

        if done:
            if self.timeout:
                self.cmd_vel_publisher.publish(Twist())
            else:
                rospy.loginfo("Collision!!")
                reward = -500
                self.cmd_vel_publisher.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal!!!! +1000 reward!!")
            reward = 500
            self.cmd_vel_publisher.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.get_position(True, delete=True)
            self.goal_distance = self.get_goal_distance()
            self.get_goalbox = False
            self.steps = 0

        return reward

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

        for action in self.past_action:
            scan_range.append(action)

        if self.steps > self.cfg["max_steps_per_episode"]:
            rospy.loginfo("Time out!!")
            self.timeout = True
            done = True

        current_distance = self.get_goal_distance()
        if current_distance < 0.15:
            self.get_goalbox = True

        return (
            scan_range + [heading, current_distance],
            done,
        )

    def step(self, action: float):
        linear_vel = action[0]
        w_vel = action[1]

        self.steps += 1

        cmd_vel = Twist()
        cmd_vel.linear.x = linear_vel
        cmd_vel.angular.z = w_vel
        self.cmd_vel_publisher.publish(cmd_vel)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message("scan", LaserScan, timeout=5)
            except:
                pass

        state, done = self.get_state(data)
        reward = self.get_reward(state, done)

        self.past_action = action

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
        state, _ = self.get_state(data)

        self.steps = 0

        return np.asarray(state)


if __name__ == "__main__":
    rospy.init_node("tl_environment", anonymous=True)
    env = Env()
    env.reset()
