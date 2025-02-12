#!/usr/bin/env python3
# Authors: Lucas G. #

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import datetime
import sys
import time

import numpy as np
import rospy
import tensorflow as tf
import yaml

from nodes.ddpg_agent import DQNAgent

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.tb3_environment import Env


def drl(env: Env, agent: DQNAgent, cfg):

    start_time = time.perf_counter()

    if not os.path.isdir("keras_models"):
        print("Models folder doesn't exist. Creating it.")
        os.mkdir("keras_models")

    for episode in range(
        cfg["current_episode"] + 1, cfg["current_episode"] + cfg["num_episodes"]
    ):
        done = False
        reward_sum = 0
        observation = env.reset()

        while not done:
            action = agent.get_action(observation)
            observation_, reward, done = env.step(action)
            reward_sum += reward
            agent.append_memory(observation, action, reward, observation_, int(done))
            observation = observation_
            agent.train_model()

        with agent.summary_writer.as_default():
            tf.summary.scalar("Max reward per episode", reward_sum, step=episode)
            tf.summary.scalar("epsilon per episode", agent.epsilon, step=episode)
            tf.summary.scalar(
                "Loss per episode", np.mean(agent.loss_memory), step=episode
            )

        if episode % cfg["save_model_every"] == 0 and episode > 10:
            agent.update_target_model()
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
    rospy.set_param("log_level", log_level)
    rospy.init_node("drl_training", log_level=log_level, anonymous=True)
    rospy.loginfo(f"Log level: {log_level}")

    env = Env()
    drl_training = DQNAgent(env.state_size, env.action_size, cfg)

    drl(env, drl_training, cfg)
