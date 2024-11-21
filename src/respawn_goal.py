#!/usr/bin/env python
# Authors: Lucas G. #

import os
import random
import time

import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import Pose


class RespawnGoal(object):
    """This should work as an API to replace goal box model into Gazebo and set
    goal postion."""

    def __init__(self):
        modelPath = os.path.dirname(os.path.realpath(__file__))
        modelPath = self.modelPath.replace(
            "transfer_learning/src", "transfer_learning/models/goal_box/model.sdf"
        )

        self.goal_model = open(modelPath, "r").read()
        self.goal_position = Pose()
        self.init_goal_x = 0.6
        self.init_goal_y = 0.0
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y

        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y

        # Env variables
        self.modelName = "goal"
        self.check_model = False

        # Subscribers
        rospy.wait_for_message("gazebo/model_states", ModelStates, 4)
        self.model_subscriber = rospy.Subscriber(
            "gazebo/model_states", ModelStates, self.model_subscriber_callback
        )

    def model_subscriber_callback(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def check_model(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawn_model(self):
        while True:
            if not self.check_model:
                rospy.wait_for_service("gazebo/spawn_sdf_model")
                spawn_model_prox = rospy.ServiceProxy(
                    "gazebo/spawn_sdf_model", SpawnModel
                )
                spawn_model_prox(
                    self.modelName,
                    self.goal_model,
                    "transfer_learning",
                    self.goal_position,
                    "world",
                )
                rospy.loginfo(
                    "Goal position : %.1f, %.1f",
                    self.goal_position.position.x,
                    self.goal_position.position.y,
                )
                break
            else:
                pass

    def delete_model(self):
        while True:
            if self.check_model:
                rospy.wait_for_service("gazebo/delete_model")
                del_model_prox = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass

    def get_position(self, delete=False):
        if delete:
            self.delete_model()

        check_model = False

        goal_x_list = [0.6, 1.9, 0.5, 0.2, -0.8, -1, -1.9, 0.5, 2, 0.5, 0, -0.1, -2]
        goal_y_list = [0, -0.5, -1.9, 1.5, -0.9, 1, 1.1, -1.5, 1.5, 1.8, -1, 1.6, -0.8]

        self.index = random.randrange(0, len(goal_x_list))
        rospy.loginfo("Goal position updating to: ...")
        rospy.loginfo(
            "Index: %d, X position: %d Y position: %d",
            self.index,
            goal_x_list[self.index],
            goal_y_list[self.index],
        )
        if self.last_index == self.index:
            position_check = True
        else:
            self.last_index = self.index
            position_check = False

        self.goal_position.position.x = goal_x_list[self.index]
        self.goal_position.position.y = goal_y_list[self.index]

        time.sleep(0.5)
        self.respawnModel()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y
