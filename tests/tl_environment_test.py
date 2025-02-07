#!/usr/bin/env python3
# Authors: Lucas G. #

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import time
import unittest

import rospy
import rostest
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist

from src.tb3_environment import Env


class TestTlEnvironment(unittest.TestCase):

    def test_turtlebot_model_should_spawn(self):
        rospy.init_node("tl_environment_test")
        rospy.sleep(0.1)
        rospy.wait_for_service("/gazebo/set_physics_properties", 4)
        # rospy.wait_for_message("/cmd_vel", Twist, timeout=4)
        model_states = rospy.wait_for_message(
            "/gazebo/model_states", ModelStates, timeout=4
        )
        self.assertTrue("turtlebot3_burger" in model_states.name)

    def test_environment_should_reset(self):
        rospy.init_node("tl_environment_test")
        rospy.sleep(0.1)
        env = Env()
        env.reset()


if __name__ == "__main__":
    rostest.rosrun("transfer_learning", "tl_environment_test", TestTlEnvironment)
