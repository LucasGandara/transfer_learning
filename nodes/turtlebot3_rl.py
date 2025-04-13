#!/usr/bin/env python3
# Authors: Lucas G. #

import os
import sys

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import rospy
import tensorflow as tf
import yaml
from gazebo_msgs.srv import DeleteModel, DeleteModelRequest

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.agents import Agent, DDPGAgent, TD3Agent
from src.consts import timeit
from src.tb3_environment import Env


@timeit
def start_drl(env: Env, agent: Agent, cfg: dict):
    scores = []
    for episode in range(
        cfg["current_episode"] + 1, cfg["current_episode"] + cfg["num_episodes"]
    ):
        agent.episode_score = 0.0
        observation = env.reset()
        agent.actor_loss_memory = 0
        agent.critic_loss_memory = 0
        episode_steps = 0
        episode_actions = []

        while True:
            action = agent.get_action(
                keras.ops.expand_dims(keras.ops.convert_to_tensor(observation), 0)
            )
            episode_actions.append(action)
            observation_, reward, done = env.step(action)
            agent.episode_score += reward
            agent.store_transition(
                observation, action, reward, observation_, 1 if done else 0
            )
            agent.learn()
            agent.update_targets()
            episode_steps += 1

            if done:
                break

            observation = observation_

        scores.append(agent.episode_score)
        episode_actions = tf.stack(episode_actions)
        avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)

        rospy.loginfo(
            f"Episode: {episode}. Score: {agent.episode_score:.2f}. Avg: {avg_score:.2f}"
        )

        with agent.summary_writer.as_default():
            # Existing metrics
            tf.summary.scalar("Reward per episode", agent.episode_score, step=episode)
            tf.summary.scalar("Avg reward (past 50 episodes)", avg_score, step=episode)
            tf.summary.scalar(
                "Accumulated critic loss per episode",
                agent.critic_loss_memory,
                step=episode,
            )
            tf.summary.scalar(
                "Accumulated actor loss per episode",
                agent.actor_loss_memory,
                step=episode,
            )

            # New episode-level metrics
            tf.summary.scalar("Episode steps", episode_steps, step=episode)
            tf.summary.scalar(
                "Mean action", tf.reduce_mean(episode_actions), step=episode
            )
            tf.summary.scalar(
                "Action std dev", tf.math.reduce_std(episode_actions), step=episode
            )
            tf.summary.scalar(
                "Max action", tf.reduce_max(episode_actions), step=episode
            )
            tf.summary.scalar(
                "Min action", tf.reduce_min(episode_actions), step=episode
            )
            tf.summary.histogram(
                "Episode actions distribution", episode_actions, step=episode
            )

        if episode % cfg["save_model_every"] == 0 and episode > 10:
            agent.save_weights()

        rospy.sleep(0.1)


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


if __name__ == "__main__":
    # Load configuration
    cfg = None
    current_file_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    cfg_file_path = current_file_path + "/config/drl_config.yml"

    with open(cfg_file_path, "r") as file:
        cfg = yaml.safe_load(file)

    LOG_LEVEL = rospy.DEBUG if cfg["log_level"] == "DEBUG" else rospy.INFO

    # Folder to save models
    models_path = "{}/keras_models".format(current_file_path)
    if not os.path.isdir(models_path):
        print("Models folder doesn't exist. Creating it.")
        os.mkdir(models_path)

    # Launch node
    rospy.set_param("log_level", LOG_LEVEL)
    rospy.init_node("drl_training", log_level=LOG_LEVEL, anonymous=True)
    rospy.on_shutdown(on_shutdown)

    rospy.loginfo(f"Log level: {LOG_LEVEL}")

    env = Env(cfg)

    agent = cfg["agent"]
    rospy.loginfo(f"Using {agent} agnet!\n")

    if agent == "DDPG":
        agent = DDPGAgent(env.state_size, env.action_size, cfg)
    elif agent == "TD3":
        agent = TD3Agent(env.state_size, env.action_size, cfg)
    else:
        rospy.logerr("Invalid agent type. Exiting...")
        exit(1)

    start_drl(env, agent, cfg)
