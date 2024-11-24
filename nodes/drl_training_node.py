#!/usr/bin/env python3
# Authors: Lucas G. #

import os
import sys
import time

import numpy as np
import rospy
import yaml
from dqn_agent import DQNAgent

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.summary_writer import MetricsWriter
from src.tl_environment import Env


def drl(env: Env, agent: DQNAgent, cfg):

    summary_writer = MetricsWriter(cfg)
    start_time = time.time()

    if not os.path.isdir("keras_models"):
        print("Models folder doesn't exist. Creating it.")
        os.mkdir("keras_models")

    for episode in range(cfg["current_episode"] + 1, cfg["num_episodes"]):
        rospy.logdebug("Episode: %s, Step: %s", episode, episode)

        done = False
        state = env.reset()
        score = 0
        for _ in range(cfg["episode_step"]):
            summary_writer.step += 1
            action = agent.get_action(state)

            next_state, reward, done = env.step(action)
            agent.append_memory(state, action, reward, next_state, done)

            # Minimum memory size to start training.
            if len(agent.memory) >= cfg["batch_size"]:
                if episode <= cfg["update_target_every"]:
                    agent.train_model()
                else:
                    agent.train_model(target=True)

            score += reward
            state = next_state

            if done:
                # Log to the console:
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                rospy.loginfo(
                    "Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d",
                    episode,
                    score,
                    len(agent.memory),
                    agent.epsilon,
                    h,
                    m,
                    s,
                )
                break

            # Update target model
            if episode % cfg["update_target_every"] == 0:
                agent.update_target_model()

            # Save checkpoint models
            if episode % cfg["save_model_every"] == 0:
                file_name = f"keras_models/{cfg['model_name']}.keras"
                agent.model.save(file_name)

            # Log to tensorboard
            if episode % cfg["aggregate_stats_every"] == 0:
                summary_writer.update_stats(
                    epsilon=agent.epsilon,
                    average_reward=np.average(agent.memory[2]),
                    max_reward=np.max(agent.memory[2]),
                    min_reward=np.min(agent.memory[2]),
                    loss=np.average(agent.memory[5]),
                )

        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)


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
    rospy.init_node("drl_training", log_level=rospy.DEBUG, anonymous=True)

    env = Env()
    drl_training = DQNAgent(env.state_size, env.action_size, cfg)

    drl(env, drl_training, cfg)
