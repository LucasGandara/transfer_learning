#!/usr/bin/env python3
# Authors: Lucas G. #

import math

import numpy as np
import rospy
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion

try:
    from src.consts import get_stage, get_stage_name
    from src.respawn_goal import RespawnGoal
    from src.reward_functions import combined_reward_function
except ModuleNotFoundError:
    from consts import get_stage, get_stage_name
    from respawn_goal import RespawnGoal
    from reward_functions import combined_reward_function


class Env(object):
    def __init__(self, cfg):
        # Configuration
        self.cfg = cfg

        self.state_size = 28
        self.action_size = 1
        self.past_action = [0] * self.action_size
        self.steps = 0
        self.timeout = False
        self.get_goalbox = False
        self.init_goal = True  # First time the Goal is initialized

        # Goal
        self.goal_x = 0.0
        self.goal_y = 0.0

        # Odometry
        self.heading = 0
        self.position = Pose()

        stage = get_stage(cfg["stage"])
        rospy.loginfo("Environment stage: {}".format(get_stage_name(cfg["stage"])))
        self.respawn_goal = RespawnGoal(stage)

        # Node publisher
        self.cmd_vel_publisher = rospy.Publisher("cmd_vel", Twist, queue_size=5)
        self.reward_publisher = rospy.Publisher("reward", Float32, queue_size=5)
        self.goal_position_publisher = rospy.Publisher(
            "/current_goal_position", Float32, queue_size=5
        )

        # Node subscriptions
        self.reset_proxy = rospy.ServiceProxy("gazebo/reset_simulation", Empty)
        rospy.Subscriber("odom", Odometry, self.odom_callback)

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

        goal_distance_msg = Float32()
        goal_distance_msg.data = goal_distance
        self.goal_position_publisher.publish(goal_distance_msg)

        return goal_distance

    def get_reward(self, state, action_w, done):
        current_distance = state[-1]
        heading = abs(state[-2])

        reward = combined_reward_function(
            theta=heading,
            omega=action_w,
            start_distance=self.goal_distance,
            current_distance=current_distance,
            w_angular=self.cfg["w_angular"],
            w_distance=self.cfg["w_distance"],
            alpha=self.cfg["alpha_reward"],
        )

        # reward = reward * (1 - step / self.cfg["max_steps_per_episode"])

        if done:
            # if self.timeout:
            #     self.cmd_vel_publisher.publish(Twist())
            #     reward = -150
            # else:
            rospy.loginfo("Collision!!")
            reward = -150
            self.cmd_vel_publisher.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal!!!! +1000 reward!!")
            reward = 100
            self.cmd_vel_publisher.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.get_position(True, delete=True)
            self.goal_distance = self.get_goal_distance()
            self.get_goalbox = False
            self.steps = 0

        reward_msg = Float32()
        reward_msg.data = reward
        self.reward_publisher.publish(reward_msg)

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

        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)

        if min_range > min(scan_range) > 0:
            done = True

        # for action in self.past_action:
        #     scan_range.append(action)

        # if self.steps > self.cfg["max_steps_per_episode"]:
        #     rospy.loginfo("Time out!!")
        #     self.timeout = True
        #     done = True

        current_distance = self.get_goal_distance()
        if current_distance < 0.15:
            self.get_goalbox = True

        return (
            scan_range
            + [obstacle_min_range, obstacle_angle, heading, current_distance],
            done,
        )

    def step(self, action: float):
        linear_vel = 0.15
        w_vel = action[0]

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
        reward = self.get_reward(state, action, done)

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
    import os

    import yaml
    from std_msgs.msg import Float32

    # Load configuration
    cfg = None
    current_file_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    cfg_file_path = current_file_path + "/config/drl_config.yml"

    with open(cfg_file_path, "r") as file:
        cfg = yaml.safe_load(file)

    rospy.init_node("tb3_environment", anonymous=True)

    reward_publisher = rospy.Publisher("reward", Float32, queue_size=5)

    linear_vel = 0.0
    angular_vel = 0.0

    def cmd_vel_callback(msg):
        global linear_vel, angular_vel

        linear_vel = msg.linear.x
        angular_vel = msg.angular.z

    rospy.Subscriber("cmd_vel", Twist, cmd_vel_callback)

    env = Env(cfg)
    env.reset()

    while True:
        state, reward, done = env.step([linear_vel, angular_vel])
        rospy.loginfo("Reward: {}".format(reward))
        reward_publisher.publish(reward)
        rospy.sleep(0.1)
