from agilerl.algorithms.maddpg import MADDPG
from env.multi_hydrogen_recharge import MultiHydrogenRecharge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from env.multi_hydrogen_recharge import MultiHydrogenRecharge

# Set the default parameters for running the environment simulation
seed = 30
num_vehicles = 5

# ---------------------------------------------------------------------

np.random.seed(seed)

# Load the built algorithm
checkpoint_path = "maddpg_agent"
agent = MADDPG.load(checkpoint_path)

# Stores the rewards and number of commands
agent.scores = []
num_commands_list = []
num_steps_list = []

env = MultiHydrogenRecharge(num_vehicles=num_vehicles, seed=seed)

# Define the parameters for testing the algorithm
episodes = 200
max_steps = 15
avg_after_episodes = 200

for ep in range(episodes):
    state = env.reset() # Reset environment at start of episode
    agent_reward = {i: 0 for i in range(env.num_vehicles)}
    num_commands = 0
    num_steps = 0	

    for _ in range(max_steps):

        # Get next action from agent
        cont_actions, discrete_action = agent.getAction(
            state
        )
        if agent.discrete_actions:
            action = discrete_action
        else:
            action = cont_actions

        next_state, reward, done = env.step(
            action
        )  # Act in environment

        for i, r in enumerate(reward):
            agent_reward[i] += r

        # Update the state
        state = next_state

        num_commands = env.num_commands
        num_commands_list.append(num_commands)
        num_steps += 1

        # Stop episode if all agents have terminated
        if all(done.values()):
            break
        
    # Save the total episode reward and number of commands
    score = sum(agent_reward.values())
    agent.scores.append(score)
    num_steps_list.append(num_steps)

    print('Actual Episode:', ep, '/ Reward:', score, '/ Number of Commands:', num_commands)

    # Print average reward of the last 200 episodes
    if ep % avg_after_episodes == 0 and ep != 0:
        avg_last_200 = np.mean(agent.scores)
        print(f'Episode: {ep}, Average Reward: {avg_last_200}')

# ------------------------------------------------------------------------

# List to store the average rewards and commands every 200 episodes
avg_rewards = []

# Total number of episodes
total_episodes = len(agent.scores)

# Calculating average rewards and commands every 200 episodes
for ep in range(200, total_episodes+1, avg_after_episodes):
    avg_last_200_rewards = np.mean(agent.scores[0:ep])
    avg_rewards.append(avg_last_200_rewards)
    print(f'Episode: {ep}, Average Reward: {avg_last_200_rewards}')

# Calculate the standard deviation of rewards every 200 episodes
std_rewards = np.std(avg_rewards)

# Plot the graph of reward averages with standard deviation
fig, ax1 = plt.subplots(figsize=(14, 6))

color = 'tab:blue'
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Reward Average', color=color)
ax1.plot(range(avg_after_episodes, total_episodes + 1, avg_after_episodes), avg_rewards, marker='o', linestyle='-', color=color)
ax1.fill_between(range(avg_after_episodes, total_episodes + 1, avg_after_episodes),
                 np.array(avg_rewards) - np.array(std_rewards),
                 np.array(avg_rewards) + np.array(std_rewards),
                 color=color, alpha=0.2)
ax1.tick_params(axis='y', labelcolor=color)

plt.title("Average Reward Over Episodes To Test The Algorithm")
fig.tight_layout()
plt.grid(True)
plt.show()

# Plot the graph of number of commands
fig, ax2 = plt.subplots(figsize=(30, 12))

color = 'tab:red'
ax2.set_xlabel('Steps')
ax2.set_ylabel('Number of Commands', color=color)
ax2.plot(range(1, 100 + 1), num_commands_list[:100], marker='x', linestyle='--', color=color)
ax2.axhline(y=num_vehicles, color='green', linestyle='-', label='Number of Vehicles')
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Number of Commands Over The First 5 Episodes To Test The Algorithm")
fig.tight_layout()
plt.grid(True)
plt.legend()
plt.show()

print('Descriptive statistics of the number of steps')
print(pd.Series(num_steps_list).describe())

# Show the table of descriptive statistics of average rewards
print('Descriptive statistics of average rewards')
print(pd.Series(avg_rewards).describe())