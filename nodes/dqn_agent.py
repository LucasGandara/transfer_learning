#!/usr/bin/env python3
# Authors: Lucas G. #

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
os.environ["KERAS_BACKEND"] = "torch"

import random
from collections import deque

import keras
import numpy as np

from src.summary_writer import MetricsWriter


# TODO: Delete Goal on shutdown
class DQNAgent(object):
    def __init__(
        self, state_size: int, action_size: int, cfg, summary_writer: MetricsWriter
    ):
        self.cfg = cfg

        # Logs
        self.summary_writer = summary_writer

        # RL Agent
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate: float = self.cfg["learning_rate"]
        self.batch_size = self.cfg["batch_size"]
        self.epsilon = self.cfg["epsilon"]
        self.epsilon_decay = self.cfg["epsilon_decay"]
        self.epsilon_min = self.cfg["epsilon_min"]
        self.memory = deque(maxlen=1000000)
        self.metrics = None

        self.model = self.build_model("off_model")
        self.target_model = self.build_model("target_model")

        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self, name: str):
        model = keras.models.Sequential(
            [
                keras.layers.Dense(
                    64,
                    input_shape=(self.state_size,),
                    activation="relu",
                    kernel_initializer="lecun_uniform",
                ),
                keras.layers.Dense(
                    64, activation="relu", kernel_initializer="lecun_uniform"
                ),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(
                    self.action_size, kernel_initializer="lecun_uniform"
                ),
                keras.layers.Activation("linear"),
            ],
            name=f"DQN_{name}",
        )

        model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.RMSprop(
                learning_rate=self.learning_rate, rho=0.9, epsilon=1e-06
            ),
        )
        model.summary()

        return model

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state.reshape(1, len(state)))
            self.q_value = q_value
            return np.argmax(q_value[0])

    def append_memory(self, state, action, reward, next_state, done, loss):
        self.memory.append((state, action, reward, next_state, done, loss))

    def train_model(self, target=False):
        mini_batch = random.sample(self.memory, self.batch_size)
        X_batch = np.empty((0, self.state_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_size), dtype=np.float64)

        for i in range(self.batch_size):
            states = mini_batch[i][0]
            actions = mini_batch[i][1]
            rewards = mini_batch[i][2]
            next_states = mini_batch[i][3]
            dones = mini_batch[i][4]

            self.q_value = self.model.predict(states.reshape(1, len(states)))

            if target:
                next_target = self.target_model.predict(
                    next_states.reshape(1, len(next_states))
                )

            else:
                next_target = self.model.predict(
                    next_states.reshape(1, len(next_states))
                )

            next_q_value = self.getQvalue(rewards, next_target, dones)

            X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
            Y_sample = self.q_value.copy()

            Y_sample[0][actions] = next_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

            if dones:
                X_batch = np.append(X_batch, np.array([next_states.copy()]), axis=0)
                Y_batch = np.append(
                    Y_batch, np.array([[rewards] * self.action_size]), axis=0
                )

        metrics = self.model.fit(
            X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0
        )
