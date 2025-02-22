#!/usr/bin/env python3
# Authors: Lucas G. #

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import datetime
import sys
import time

import keras
import numpy as np
import rospy
import tensorflow as tf
import yaml
from ddpg_agent import DDPGAgent
from gazebo_msgs.srv import DeleteModel, DeleteModelRequest

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from std_msgs.msg import Float64

from src.tb3_environment import Env


def on_shutdown():
    rospy.loginfo("Stopping training...")
    rospy.wait_for_service("/gazebo/delete_model", 1)

    delete_model_request = DeleteModelRequest()
    delete_model_request.model_name = "goal"

    try:
        delete_model_srv = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
        response = delete_model_srv(delete_model_request)
        rospy.loginfo("Goal model deleted.")
        rospy.loginfo(response)
    except rospy.ServiceException:
        rospy.loginfo("Failed to delete goal model.")

    rospy.loginfo("Stop training successfully!")


def ddpg(env: Env, agent: DDPGAgent, cfg):

    reward_publisher = rospy.Publisher("/reward", Float64, queue_size=1)

    start_time = time.perf_counter()

    if not os.path.isdir("keras_models"):
        print("Models folder doesn't exist. Creating it.")
        os.mkdir("keras_models")

    scores = []

    for episode in range(
        cfg["current_episode"] + 1, cfg["current_episode"] + cfg["num_episodes"]
    ):
        score = 0
        steps = 0
        observation = env.reset()
        agent.actor_loss_memory = 0
        agent.critic_loss_memory = 0

        while True:
            steps += 1
            action = agent.choose_action(
                keras.ops.expand_dims(keras.ops.convert_to_tensor(observation), 0),
                training=True,
            )

            observation_, reward, done = env.step(action)
            score += reward
            agent.append_memory(observation, action, reward, observation_)

            agent.learn()
            agent.update_targets()

            if steps >= 500:
                done = True

            if done:
                break

            observation = observation_

        scores.append(score)

        reward_publisher.publish(score)

        avg_score = np.mean(scores)
        rospy.loginfo("Episode: {}. Avg: {}".format(episode, avg_score))

        with agent.summary_writer.as_default():
            tf.summary.scalar("Reward per episode", score, step=episode)
            tf.summary.scalar("Avg_reward (past 50 episodes)", avg_score, step=episode)
            tf.summary.scalar(
                "Accumulated critic_loss per episode",
                agent.critic_loss_memory,
                step=episode,
            )
            tf.summary.scalar(
                "Accumulated actor_loss per episode",
                agent.actor_loss_memory,
                step=episode,
            )

        if episode % cfg["save_model_every"] == 0 and episode > 10:
            agent.save_model()

    end = time.perf_counter()
    duration = end - start_time
    print("Training took: ")
    print(str(datetime.timedelta(seconds=duration)))


if __name__ == "__main__":
    cfg = None
    cfg_file_path = (
        os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        + "/config/drl_config.yml"
    )

    with open(cfg_file_path, "r") as file:
        cfg = yaml.safe_load(file)

    log_level = rospy.DEBUG if cfg["log_level"] == "DEBUG" else rospy.INFO
    rospy.init_node("drl_training", log_level=log_level, anonymous=True)

    rospy.set_param("log_level", log_level)
    rospy.loginfo(f"Log level: {log_level}")
    rospy.on_shutdown(on_shutdown)

    env = Env()
    agent = DDPGAgent(env.state_size, env.action_size, cfg)

    ddpg(env, agent, cfg)
