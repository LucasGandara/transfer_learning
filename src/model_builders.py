import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras


def build_actor(n_states: int, upper_bound_action: int, name: str) -> keras.Model:
    # Initialize weights between -3e-3 and 3-e3
    last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

    inputs = keras.layers.Input(shape=(n_states,), name="input_layer")
    out = keras.layers.Dense(256, activation="relu", name="dense_1")(inputs)
    out = keras.layers.Dense(256, activation="relu", name="dense_2")(out)
    outputs = keras.layers.Dense(
        1, activation="tanh", kernel_initializer=last_init, name="output_layer"
    )(out)

    # Our upper bound is 1.5 for robot velocity.
    outputs = outputs * upper_bound_action
    model = keras.Model(inputs, outputs, name="name")
    return model


def build_critic(
    state_dim: int,
    n_actions: int,
    name: str,
) -> keras.Model:
    ### Critic on the state - action pair. It should return a value of the pair
    state_input = keras.layers.Input(shape=(state_dim,), name="state_input")
    state_out = keras.layers.Dense(16, activation="relu", name="state_out")(state_input)
    state_value = keras.layers.Dense(32, activation="relu", name="state_value")(
        state_out
    )

    action_input = keras.layers.Input(shape=(n_actions,), name="actin_input")
    action_value = keras.layers.Dense(32, activation="relu", name="action_value")(
        action_input
    )

    intermediate = keras.layers.Concatenate()([state_value, action_value])

    out = keras.layers.Dense(256, activation="relu")(intermediate)
    out = keras.layers.Dense(256, activation="relu")(out)
    outputs = keras.layers.Dense(1, activation=None, name="critic_value")(out)

    model = keras.Model([state_input, action_input], outputs, name=name)

    return model
