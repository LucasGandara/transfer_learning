# Author: Lucas G.

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import tensorflow as tf


# Fan-In initialization function
def fanin_init(shape, dtype=None, scale=1.0):
    fan_in = shape[0]  # Number of input units (fan-in)
    limit = scale * np.sqrt(1.0 / fan_in)
    return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype)


# Custom Keras layer with Fan-In initializer
class FanInDense(keras.layers.Layer):
    def __init__(self, units, activation=None, scale=1.0, name=""):
        super(FanInDense, self).__init__(name=name)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.scale = scale

    def build(self, input_shape):
        # Initialize weights with Fan-In initialization
        fan_in = input_shape[-1]
        self.w = self.add_weight(
            shape=(fan_in, self.units),
            initializer=lambda shape, dtype: fanin_init(shape, dtype, scale=self.scale),
            trainable=True,
            name="kernel",
        )
        # Initialize bias to zeros
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True, name="bias"
        )

    def call(self, inputs):
        # Forward pass
        z = tf.matmul(inputs, self.w) + self.b
        if self.activation is not None:
            return self.activation(z)
        return z


class Actor(keras.Model):
    def __init__(self, action_dim, action_limit_v, action_limit_w, name):
        super(Actor, self).__init__(name=name)
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w

        self.dense_1 = FanInDense(500, activation="relu", scale=1.0, name="dense1")
        self.dense_2 = FanInDense(300, activation="relu", scale=1.0, name="dense2")
        self.dense_3 = keras.layers.Dense(
            action_dim,
            kernel_initializer=keras.initializers.RandomUniform(
                minval=-0.003, maxval=0.003
            ),
            name="action",
        )

    def call(self, state):
        x = self.dense_1(state)
        x = self.dense_2(x)
        actions = self.dense_3(x)

        # Linear and angular velocities
        action_v = tf.sigmoid(actions[..., 0:1]) * self.action_limit_v
        action_w = tf.tanh(actions[..., 1:2]) * self.action_limit_w

        return tf.concat([action_v, action_w], axis=-1)


class Critic(keras.Model):
    def __init__(self, name):
        super(Critic, self).__init__(name=name)

        self.state_value_layer = FanInDense(
            250, activation="relu", scale=1.0, name="state_value"
        )

        self.action_value_layer = FanInDense(
            250, activation="relu", scale=1.0, name="action_value"
        )

        # Combined pathway
        self.concat_layer = keras.layers.Concatenate()
        self.intermediate_layer = FanInDense(
            250, activation="relu", scale=1.0, name="intermediate"
        )

        # Output layer for the critic value
        self.output_layer = keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=keras.initializers.RandomUniform(-0.003, 0.003),
            name="critic_value",
        )

    def call(self, inputs):
        state_input, action_input = inputs

        # Process state input
        state_value = self.state_value_layer(state_input)

        # Process action input
        action_value = self.action_value_layer(action_input)

        # Concatenate state and action paths
        combined = self.concat_layer([state_value, action_value])

        # Intermediate and output layers
        intermediate = self.intermediate_layer(combined)
        critic_value = self.output_layer(intermediate)

        return critic_value


if __name__ == "__main__":
    # Example usage:
    state_dim = 28  # Example state dimension
    n_actions = 2  # Example number of actions

    critic = Critic(name="ddpg_critic")

    # Example inputs for state and action
    state_input = tf.random.uniform(
        (1, state_dim)
    )  # Batch size of 1, random state input
    action_input = tf.random.uniform(
        (1, n_actions)
    )  # Batch size of 1, random action input

    # Print model summary
    critic.build([(None, state_dim), (None, n_actions)])
    # Call the model with inputs
    critic_value = critic([state_input, action_input])
    critic.summary()

    print("Critic Value:", critic_value)

    actor = Actor(state_dim, n_actions, 1.0, 1.0, name="ddpg_actor")

    # Example inputs for state
    state_input = tf.random.uniform(
        (1, state_dim)
    )  # Batch size of 1, random state input
    # Call the model with inputs
    actions = actor(state_input)
    actor.summary()
