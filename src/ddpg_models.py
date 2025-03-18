# Author: Lucas G.

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf


# Custom Dense Layer with Fan-In Initialization
class FanInDense(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(FanInDense, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        fan_in = input_shape[-1]  # Number of input units
        # Fan-in weight initialization: Variance scaling
        initializer = keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_in", distribution="untruncated_normal"
        )
        self.w = self.add_weight(
            shape=(fan_in, self.units),
            initializer=initializer,
            trainable=True,
            name="kernel",
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True, name="bias"
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


def create_actor_model(input_shape, max_w, max_v, name):
    ### Actor network. It should return a value for each action
    inputs = keras.layers.Input(shape=input_shape, name="input")
    out = keras.layers.Dense(256, activation="relu", name="dense_1")(inputs)
    out = keras.layers.Dense(256, activation="relu", name="dense_2")(out)
    outputs = keras.layers.Dense(
        1,
        activation="tanh",
        kernel_initializer=keras.initializers.RandomUniform(
            minval=-0.003, maxval=0.003
        ),  # Kernel initializer given by the paper (because of the activation function)
        name="output",
    )(out)

    outputs = outputs * max_w  # 2 is the max value the action can take: see below link
    # https://gymnasium.farama.org/environments/classic_control/pendulum/

    model = keras.Model(inputs, outputs, name=name)

    return model


# Critic model
def create_critic_model(state_input_shape, action_input_shape, name):
    ### Critic on the state - action pair. It should return a value of the pair
    state_input = keras.layers.Input(shape=state_input_shape, name="state_input")
    state_out = keras.layers.Dense(16, activation="relu", name="state_out")(state_input)
    state_value = keras.layers.Dense(32, activation="relu", name="state_value")(
        state_out
    )

    action_input = keras.layers.Input(shape=action_input_shape, name="actin_input")
    action_value = keras.layers.Dense(32, activation="relu", name="action_value")(
        action_input
    )

    intermediate = keras.layers.Concatenate()([state_value, action_value])

    out = keras.layers.Dense(256, activation="relu")(intermediate)
    out = keras.layers.Dense(556, activation="relu")(out)
    outputs = keras.layers.Dense(1, activation=None, name="critic_value")(out)

    model = keras.Model([state_input, action_input], outputs, name=name)

    return model


if __name__ == "__main__":
    # Example usage:
    state_dim = (28,)  # Example state dimension
    action_dim = (2,)  # Example number of actions

    critic = create_critic_model(state_dim, action_dim, name="Critic_model")

    # Example inputs for state and action
    state_input = tf.random.uniform(
        (1, *state_dim)
    )  # Batch size of 1, random state input
    action_input = tf.random.uniform(
        (1, *action_dim)
    )  # Batch size of 1, random action input

    # Call the model with inputs
    critic_value = critic([state_input, action_input])
    critic.summary()

    print("Critic Value:", critic_value)

    actor = actor_model = create_actor_model(state_dim, 100, 100, name="Actor_model")

    # Example inputs for state
    state_input = tf.random.uniform(
        (10, *state_dim)
    )  # Batch size of 1, random state input
    # Call the model with inputs
    actions = actor(state_input)
    actor.summary()

    print(actions)
