import gym
import numpy as np

# Define the CartPole environment
env = gym.make('CartPole-v1')

# Define Q-learning parameters
num_episodes = 1000
num_bins = (1, 1, 6, 12)  # Number of bins for each observation dimension
num_actions = env.action_space.n  # Number of possible actions
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial epsilon for epsilon-greedy policy
epsilon_decay = 0.995  # Decay rate of epsilon
epsilon_min = 0.01  # Minimum value of epsilon

# Initialize Q-table
Q = np.zeros(num_bins + (num_actions,))

# Define observation discretization function
def discretize(observation):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], np.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -np.radians(50)]
    ratios = [(observation[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(observation))]
    discretized_observation = [int(np.digitize(ratios[i], np.linspace(0, 1, num_bins[i]))) for i in range(len(observation))]
    discretized_observation = [min(num_bins[i] - 1, max(0, discretized_observation[i] - 1)) for i in range(len(observation))]
    return tuple(discretized_observation)

# Define epsilon-greedy policy
def epsilon_greedy_policy(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()  # Random action
    else:
        return np.argmax(Q[state])  # Greedy action

# Train the Q-learning agent
for episode in range(num_episodes):
    observation, info = env.reset()
    state = discretize(observation)
    done = False
    total_reward = 0

    while not done:
        action = epsilon_greedy_policy(state, epsilon)
        next_observation, reward, done, _ = env.step(action)
        next_state = discretize(next_observation)

        # Update Q-table
        Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

        state = next_state
        total_reward += reward

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Print episode statistics
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Test the trained agent
total_rewards = []
for _ in range(100):
    observation, info = env.reset()
    state = discretize(observation)
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(Q[state])
        observation, reward, done, _ = env.step(action)
        state = discretize(observation)
        total_reward += reward

    total_rewards.append(total_reward)

# Print average reward over 100 episodes
print(f"Average Reward over 100 Episodes: {np.mean(total_rewards)}")

# Close the environment
env.close()
