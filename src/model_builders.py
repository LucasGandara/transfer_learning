import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf


class Actor(keras.Model):
    def __init__(self, state_dim, action_dim, action_limit_v, action_limit_w, name):
        super(Actor, self).__init__(name=name)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w

        # Custom initializer to match fanin_init behavior
        initializer = keras.initializers.VarianceScaling(
            scale=1.0 / 3.0, mode="fan_in", distribution="uniform"
        )

        # Layer definitions
        self.fa1 = keras.layers.Dense(
            500, activation="relu", kernel_initializer=initializer
        )

        self.fa2 = keras.layers.Dense(
            500, activation="relu", kernel_initializer=initializer
        )

        self.fa3 = keras.layers.Dense(
            action_dim,
            kernel_initializer=keras.initializers.RandomUniform(
                minval=-0.003, maxval=0.003
            ),
        )

    def call(self, state):
        x = self.fa1(state)
        x = self.fa2(x)
        action = self.fa3(x)

        # Batch case
        action_v = tf.sigmoid(action[..., 0:1]) * self.action_limit_v
        action_w = tf.tanh(action[..., 1:2]) * self.action_limit_w

        return tf.concat([action_v, action_w], axis=-1)


def build_actor(
    n_states: int, upper_bound_action_v: float, upper_bound_action_w: float, name: str
) -> keras.Model:
    model = Actor(n_states, 2, upper_bound_action_v, upper_bound_action_w, name)
    return model


def build_critic(
    state_dim: int,
    n_actions: int,
    name: str,
) -> keras.Model:
    ### Critic on the state - action pair. It should return a value of the pair

    initializer = keras.initializers.VarianceScaling(
        scale=1.0 / 3.0, mode="fan_in", distribution="uniform"
    )

    state_input = keras.layers.Input(shape=(state_dim,), name="state_input")
    state_value = keras.layers.Dense(
        250, activation="relu", kernel_initializer=initializer, name="state_value"
    )(state_input)

    action_input = keras.layers.Input(shape=(n_actions,), name="action_input")
    action_value = keras.layers.Dense(
        250, activation="relu", kernel_initializer=initializer, name="action_value"
    )(action_input)

    intermediate = keras.layers.Concatenate()([state_value, action_value])

    out = keras.layers.Dense(
        500, activation="relu", kernel_initializer=initializer, name="intermediate"
    )(intermediate)
    outputs = keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=keras.initializers.RandomUniform(-0.003, 0.003),
        name="critic_value",
    )(out)

    model = keras.Model([state_input, action_input], outputs, name=name)

    return model
