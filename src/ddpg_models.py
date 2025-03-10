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


def create_actor_model(input_shape, action_limit_v, action_limit_w, name):
    inputs = keras.layers.Input(shape=input_shape, name="states")
    dense_1 = FanInDense(500, name="fan_in_dense_1")(inputs)
    dense_2 = FanInDense(500, name="fan_in_dense_2")(dense_1)

    # Output 1: Angular Velocity (example: Dense layer with one unit)
    angular_velocity = keras.layers.Dense(
        1, activation="tanh", name="angular_velocity"
    )(dense_2)
    angular_velocity = keras.layers.Lambda(
        lambda x: x * action_limit_w, name="angular_scaling"
    )(angular_velocity)

    # Output 2: Linear Velocity (example: Dense layer with one unit)
    linear_velocity = keras.layers.Dense(
        1, activation="sigmoid", name="linear_velocity"
    )(dense_2)
    linear_velocity = keras.layers.Lambda(
        lambda x: x * action_limit_v, name="linear_scaling"
    )(linear_velocity)

    # Model with two outputs
    model = keras.models.Model(
        inputs=inputs, outputs=[angular_velocity, linear_velocity], name=name
    )
    return model


# Critic model
def create_critic_model(state_input_shape, action_input_shape, name):
    state_input = keras.layers.Input(shape=state_input_shape, name="states")
    state_value = FanInDense(250, name="state_value")(state_input)

    action_input = keras.layers.Input(shape=action_input_shape, name="actions")
    action_value = FanInDense(250, name="action_value")(action_input)

    combined = keras.layers.Concatenate(name="combine")([state_value, action_value])
    intermediate = FanInDense(250, name="intermediate")(combined)

    critic_value = keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=keras.initializers.RandomUniform(-0.003, 0.003),
        name="critic_value",
    )(intermediate)

    model = keras.models.Model(
        inputs=[state_input, action_input], outputs=critic_value, name=name
    )

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

    actor = actor_model = create_actor_model(state_dim, 2.2, 1.5, name="Actor_model")

    # Example inputs for state
    state_input = tf.random.uniform(
        (1, *state_dim)
    )  # Batch size of 1, random state input
    # Call the model with inputs
    actions = actor(state_input)
    actor.summary()
