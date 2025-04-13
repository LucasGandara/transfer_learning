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
    theta: float,
    omega: float,
    start_distance: float,
    current_distance: float,
    w_angular=1.0,
    w_distance=1.0,
    alpha=1.0,
):
    """Calculate the combined reward based on angular velocity, heading, and distance.

    The reward function combines multiple factors to guide robot navigation:
    - Angular components (heading and velocity)
    - Distance improvement towards the goal
    - Time penalty to encourage faster completion

    Args:
        theta (float): Heading angle in radians (difference between robot
            orientation and goal direction)
        omega (float): Angular velocity (rate of change of the robot's heading)
        start_distance (float): Distance from the starting position to the goal
        current_distance (float): Current distance from robot's position to goal
        w_angular (float, optional): Weight for angular velocity and heading
            reward. Defaults to 1.0
        w_distance (float, optional): Weight for distance-based reward.
            Defaults to 1.0
        alpha (float, optional): Weighting factor for heading improvement.
            Defaults to 1.0

    Returns:
        float: The combined reward considering heading/velocity, distance and time
    """
    # Angular reward
    heading_reward = -abs(theta)
    improvement_reward = np.sign(-theta) * np.sign(omega)
    angular_reward = heading_reward + alpha * improvement_reward

    # Distance reward (linear reward)
    distance_reward = start_distance - current_distance

    # Time penalty - scales with distance to encourage faster movement
    time_penalty = -0.1 * current_distance

    # Combined reward with time penalty
    total_reward = (
        w_angular * angular_reward + w_distance * distance_reward + time_penalty
    )

    return float(total_reward)
