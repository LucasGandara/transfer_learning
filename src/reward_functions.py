import numpy as np


def angular_reward_function(theta, omega, alpha=1):
    """
    Calculate the reward based on heading angle (theta) and angular velocity (omega).

    Parameters:
    - theta: Heading angle in radians (difference between robot orientation and goal direction).
    - omega: Angular velocity (rate of change of the robot's heading).
    - alpha: Weighting factor for the improvement reward (default is 1).

    Returns:
    - reward: The total reward based on the heading and the angular velocity.
    """
    # Reward for the absolute heading angle (penalty for being far from goal)
    heading_reward = -abs(theta)

    # Reward for reducing the heading (reward when the robot turns in the right direction)
    improvement_reward = np.sign(-theta) * np.sign(omega)

    # Total reward combines the heading and improvement rewards
    total_reward = heading_reward + alpha * improvement_reward

    return total_reward


def linear_reward_function(start_distance, current_distance):
    """
    Calculate the reward based on the robot's current distance to the goal.

    Parameters:
    - start_distance: The distance from the starting position to the goal.
    - current_distance: The current distance from the robot's position to the goal.

    Returns:
    - reward: The linear reward based on how much closer the robot is to the goal.
    """
    # Reward is the difference between the starting distance and the current distance
    reward = start_distance - current_distance

    return reward


def combined_reward_function(
    theta,
    omega,
    start_distance,
    current_distance,
    w_angular=1.0,
    w_distance=1.0,
    alpha=1.0,
):
    """
    Calculate the combined reward based on angular velocity, heading, and distance to the goal.

    Parameters:
    - theta: Heading angle in radians (difference between robot orientation and goal direction).
    - omega: Angular velocity (rate of change of the robot's heading).
    - start_distance: Distance from the starting position to the goal.
    - current_distance: Current distance from the robot's position to the goal.
    - w_angular: Weight for the angular velocity and heading reward (default is 1.0).
    - w_distance: Weight for the distance-based reward (default is 1.0).
    - alpha: Weighting factor for the improvement in heading (default is 1.0).

    Returns:
    - total_reward: The combined reward considering both heading/velocity and distance.
    """
    # Angular reward (from previous function)
    heading_reward = -abs(theta)
    improvement_reward = np.sign(-theta) * np.sign(omega)
    angular_reward = heading_reward + alpha * improvement_reward

    # Distance reward (linear reward)
    distance_reward = start_distance - current_distance

    # Combined reward
    total_reward = w_angular * angular_reward + w_distance * distance_reward

    return total_reward
