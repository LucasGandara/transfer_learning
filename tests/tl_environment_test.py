#!/usr/bin/env python
# Authors: Lucas G. #

import time
import unittest

import rospy
import rostest
from gazebo_msgs.msg import ModelStates


class TestTlEnvironment(unittest.TestCase):

    def test_turtlebot_model_should_spawn(self):
        rospy.init_node("tl_environment_test")
        rospy.wait_for_service("/gazebo/set_physics_properties", 4)
        rospy.wait_for_message("/cmd_vel", timeout=4)
        model_states = rospy.wait_for_message("/gazebo/model_states", ModelStates)
        assert "turtlebot3_burger" in model_states.name


if __name__ == "__main__":
    rostest.rosrun("transfer_learning", "tl_environment_test", TestTlEnvironment)
