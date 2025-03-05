#!/usr/bin/env python3
# Authors: Lucas G. #

import os
import sys

os.environ["KERAS_BACKEND"] = "tensorflow"
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import datetime

import keras
import matplotlib.image as mpimg
import tensorflow as tf

from src.model_builders import build_actor, build_critic
from src.replay_buffer import ReplayBuffer

print("Keras version:", keras.__version__)
print("Keras backend:", keras.backend.backend())  # tensorflow
assert keras.backend.backend() == "tensorflow"
print("TensorFlow version:", tf.__version__)


class DDPGAgent(object):
    def __init__(self, state_size: int, action_size: int, cfg):
        self.cfg = cfg

        # Logs
        self.base_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        date_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.log_dir = f"{self.base_path}/{self.cfg['base_log_dir']}/ddpg_{date_time}"
        print(f"Logging metrics to {self.log_dir}")
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.model_plotted = False

        # RL Agent
        self.state_size = state_size
        self.action_size = action_size
        self.noise_std_deviation = self.cfg["std_dev"]
        self.critic_lr = self.cfg["critic_lr"]
        self.actor_lr = self.cfg["actor_lr"]
        self.batch_size = self.cfg["batch_size"]
        self.gamma = self.cfg["gamma"]
        self.tau = self.cfg["tau"]
        self.memory = ReplayBuffer(
            self.state_size,
            self.action_size,
            self.cfg["memory_size"],
            self.cfg["batch_size"],
        )
        self.train_epochs = 0
        self.actor_loss_memory = 0
        self.critic_loss_memory = 0

        self.actor_optimizer = keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=self.critic_lr)

        if self.cfg["load_model"]:
            self.actor = keras.saving.load_model(
                f"{self.base_path}/keras_models/ddpg_actor.keras"
            )
            self.critic = keras.saving.load_model(
                f"{self.base_path}/keras_models/ddpg_actor.keras"
            )

            self.target_actor = build_actor(
                self.state_size,
                self.cfg["max_linear_vel"],
                self.cfg["max_angular_vel"],
                "target_actor_model",
            )
            self.target_critic = build_critic(
                self.state_size, self.action_size, "target_critic_model"
            )

            # Make the weights equal initially
            self.target_actor.set_weights(self.actor.get_weights())
            self.target_critic.set_weights(self.critic.get_weights())
        else:
            self.actor = build_actor(
                self.state_size,
                self.cfg["max_linear_vel"],
                self.cfg["max_angular_vel"],
                "target_actor_model",
            )
            self.critic = build_critic(
                self.state_size, self.action_size, "critic_model"
            )

            self.target_actor = self.target_actor = build_actor(
                self.state_size,
                self.cfg["max_linear_vel"],
                self.cfg["max_angular_vel"],
                "target_actor_model",
            )
            self.target_critic = build_critic(
                self.state_size, self.action_size, "target_critic_model"
            )

            # Make the weights equal initially
            self.target_actor.set_weights(self.actor.get_weights())
            self.target_critic.set_weights(self.critic.get_weights())

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self):
        self.target_actor.save(f"{self.base_path}/keras_models/ddpg_actor.keras")
        self.target_critic.save(
            f"{self.base_path}/keras_models/ddpg_critic_model.keras"
        )
        if not self.model_plotted:
            self.plot_models()
            self.model_plotted = True

    def choose_action(self, state, training=True):
        action = self.actor(state, training=training)
        # Paper states to use Ornstein-Uhlenbeck noise  (https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)
        # White noise is enough
        noise = tf.random.normal((self.action_size,), 0.0, self.cfg["std_dev"])
        noisy_action = action + noise
        noisy_action = tf.clip_by_value(
            noisy_action, -self.cfg["max_angular_vel"], self.cfg["max_angular_vel"]
        )

        return noisy_action[0]

    def append_memory(self, state, action, reward, new_state):
        self.memory.store_transition(state, action, reward, new_state)

    @tf.function
    def train_step(self, state, action, reward, new_state):
        # Critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_state, training=True)
            y = reward + self.gamma * self.target_critic(
                [new_state, target_actions], training=True
            )
            critic_value = self.critic([state, action], training=True)
            critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        # Actor
        with tf.GradientTape() as tape:
            actions = self.actor(state, training=True)
            critic_value = self.critic([state, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -keras.ops.mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

        return actor_loss, critic_loss

    def learn(self):
        if self.memory.buffer_counter > self.batch_size:
            state, action, reward, new_state = self.memory.sample_buffer(
                self.batch_size
            )

            critic_loss, actor_loss = self.train_step(state, action, reward, new_state)

            self.actor_loss_memory += actor_loss.numpy()
            self.critic_loss_memory += critic_loss.numpy()

    def update_targets(self):
        """
        This update target networks slowly
        """

        # Target actor update
        actor_target_weights = self.target_actor.get_weights()
        actor_current_weights = self.actor.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = (
                self.tau * actor_current_weights[i]
                + (1 - self.tau) * actor_target_weights[i]
            )

        self.target_actor.set_weights(actor_target_weights)

        # Target critic update
        critic_target_weights = self.target_critic.get_weights()

        critic_current_weights = self.critic.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = (
                self.tau * critic_current_weights[i]
                + (1 - self.tau) * critic_target_weights[i]
            )

        self.target_critic.set_weights(critic_target_weights)

    def plot_models(self):
        actor_model_path = (
            f"{self.base_path}/{self.cfg['model_save_path']}/actor_model.png"
        )
        critic_model_path = (
            f"{self.base_path}/{self.cfg['model_save_path']}/critic_model.png"
        )

        keras.utils.plot_model(
            self.actor,
            to_file=actor_model_path,
            show_shapes=True,
        )

        keras.utils.plot_model(
            self.critic,
            to_file=critic_model_path,
            show_shapes=True,
        )

        actor_model_img = mpimg.imread(actor_model_path)
        critic_model_img = mpimg.imread(critic_model_path)

        with self.summary_writer.as_default():
            tf.summary.image(
                "Actor model graph", tf.expand_dims(actor_model_img), step=0
            )
            tf.summary.image(
                "Critic model graph", tf.expand_dims(critic_model_img), step=0
            )
