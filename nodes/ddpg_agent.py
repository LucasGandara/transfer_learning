#!/usr/bin/env python3
# Authors: Lucas G. #

import os
import sys

os.environ["KERAS_BACKEND"] = "tensorflow"
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import datetime
from collections import deque

import keras
import numpy as np
import tensorflow as tf

print("Keras version:", keras.__version__)
print("Keras backend:", keras.backend.backend())  # tensorflow
assert keras.backend.backend() == "tensorflow"
print("TensorFlow version:", tf.__version__)


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_counter = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


# TODO: Delete Goal on shutdown
class DQNAgent(object):
    def __init__(self, state_size: int, action_size: int, cfg):
        self.cfg = cfg

        # Logs
        date_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.log_dir = f"{self.cfg['base_log_dir']}/dqn_{date_time}"
        print(f"Logging metrics to {self.log_dir}")
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

        # RL Agent
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate: float = self.cfg["learning_rate"]
        self.batch_size = self.cfg["batch_size"]
        self.epsilon = self.cfg["epsilon"]
        self.epsilon_decay = self.cfg["epsilon_decay"]
        self.min_epsilon = self.cfg["min_epsilon"]
        self.memory = ReplayBuffer(
            self.cfg["memory_size"],
            self.state_size,
            self.action_size,
            discrete=True,
        )
        self.loss_memory = deque(maxlen=1_000_000)
        self.metrics = None
        self.discount_factor = self.cfg["discount_factor"]
        self.train_epochs = 0

        if self.cfg["load_model"]:
            self.model = keras.saving.load_model(cfg["model_save_path"])
            self.target_model = keras.saving.load_model(cfg["model_save_path"])
        else:
            self.model = self.build_model("actor_model")
            self.target_model = self.build_model("target_model")
            self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self):
        self.target_model.save(self.cfg["model_save_path"])

    def build_model(self, name: str):
        model = keras.models.Sequential(
            [
                keras.layers.Dense(
                    64,
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
                keras.layers.Activation(None),
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

    def get_qvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def get_action(self, state):
        state = state[np.newaxis, :]
        random_number = np.random.random()
        if random_number <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            actions = self.model.predict(state, verbose=0)
            action = np.argmax(actions)

        return action

    def append_memory(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def train_model(self, target=False):
        loss = 0
        if self.memory.mem_counter > self.batch_size:
            self.train_epochs += 1

            state, action, reward, new_state, done = self.memory.sample_buffer(
                self.batch_size
            )

            action_values = np.array(self.action_size, dtype=np.int8)
            # action_indices = np.dot(action, action_values)
            action_indices = np.argmax(action, axis=1)

            q_eval = self.model.predict(state, verbose=0)

            q_next = self.target_model.predict(new_state, verbose=0)

            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = (
                reward + self.cfg["gamma"] * np.max(q_next, axis=1) * done
            )

            history = self.model.fit(state, q_target, verbose=0)

            self.epsilon = (
                self.epsilon * self.epsilon_decay
                if self.epsilon > self.min_epsilon
                else self.min_epsilon
            )

            loss = history.history["loss"][0]
            self.loss_memory.append(loss)

        if (
            self.train_epochs > 30
            and self.train_epochs % self.cfg["update_target_every"] == 0
        ):
            self.target_model.set_weights(self.model.get_weights())

        return loss
