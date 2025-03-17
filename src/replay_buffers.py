import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, state_size, n_actions):
        # Number of "experiences" to store at max
        self.buffer_capacity = max_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, state_size))
        self.action_buffer = np.zeros((self.buffer_capacity, n_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_size))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))

    def store_transition(self, state, action, reward, state_, done):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = state_
        self.done_buffer[index] = done

        self.buffer_counter += 1

    def sample_buffer(self, batch_size):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, batch_size)

        state_batch = keras.ops.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = keras.ops.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = keras.ops.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = keras.ops.cast(reward_batch, dtype="float32")
        next_state_batch = keras.ops.convert_to_tensor(
            self.next_state_buffer[batch_indices]
        )
        done_batch = keras.ops.cast(keras.ops.convert_to_tensor(self.done_buffer[batch_indices]), dtype="float32")

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

if __name__ == "__main__":
    import os

    import yaml
    import rospy
    from tb3_environment import Env
    from agents import DDPGAgent
    from sensor_msgs.msg import LaserScan
    import random

    # Load configuration
    cfg = None
    current_file_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    cfg_file_path = current_file_path + "/config/drl_config.yml"

    with open(cfg_file_path, "r") as file:
        cfg = yaml.safe_load(file)

    rospy.init_node("replay_buffer_test", anonymous=True)

    n_samples = 100
    publisher_list = [rospy.Publisher(f"sample_{sample}", LaserScan, queue_size=10, latch=True) for sample in range(n_samples)]

    env = Env(cfg)
    agent = DDPGAgent(env.state_size, env.action_size, cfg)
    

    observation = env.reset()

    for i in range(n_samples):
        action = agent.get_action(
            keras.ops.expand_dims(keras.ops.convert_to_tensor(observation), 0)
        )
        observation_, reward, done = env.step(action)
        agent.store_transition(observation, action, reward, observation_, done)

        observation = observation_

    intensities = [random.random() for _ in range(24)]

    while True:
        time_stamp = rospy.Time.now()
        for i in range(n_samples):
            state = agent.memory.state_buffer[i]
            ranges = state[0:24]

            laser_msg = LaserScan()
            laser_msg.header.stamp = time_stamp
            laser_msg.header.frame_id = "base_scan"
            laser_msg.angle_min = 0.0
            laser_msg.angle_max = 6.28318977355957
            laser_msg.range_min = 0.11999999731779099
            laser_msg.range_max = 3.5
            laser_msg.angle_increment = 0.008726646
            laser_msg.ranges = ranges
            laser_msg.intensities = intensities

            publisher_list[i].publish(laser_msg)
            rospy.loginfo(f"Publishing sample {i}")

        
        rospy.sleep(0.1)

