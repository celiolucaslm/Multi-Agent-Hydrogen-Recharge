from env.multi_hydrogen_recharge import MultiHydrogenRecharge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set the default parameters for running the environment simulation
seed = 42
num_vehicles = 4
num_commands = 4

# --------------------------------------------------------------------

np.random.seed(seed)

env = MultiHydrogenRecharge(num_vehicles=num_vehicles, num_commands=num_commands, seed=seed)

# Defines the test parameters for the environment's random actions
num_episodes = 10000
max_steps = 10
avg_after_episodes = 200

# Stores the rewards
reward_list = []

# External loop for episodes
for episode in range(num_episodes):
  env.reset()
  vehicle_rewards = {i: 0 for i in range(env.num_vehicles)}

  for step in range(max_steps):
      
      # Vehicles take random action
      actions = np.random.rand(env.num_vehicles, 3)
      
      # Execute the action and take the next observation, reward and done (terminal state)
      observation, rewards, done = env.step(actions)

      # After the end of the episode, keep the vehicle rewards
      for i, reward in enumerate(rewards):
          vehicle_rewards[i] += reward

  score = sum(vehicle_rewards.values())
  reward_list.append(score)
  print('Actual Episode', episode, '/ Reward: ', score)

  # Print average reward of the last 200 episodes
  if episode % avg_after_episodes == 0 and episode != 0:
        avg_last_200 = np.mean(reward_list)
        print(f'Episode: {episode}, Average Reward: {avg_last_200}')

# ------------------------------------------------------------------------------

# List to store the average rewards every 200 episodes
avg_rewards = []

# Total number of episodes
total_episodes = len(reward_list)

for ep in range(200, total_episodes+1, avg_after_episodes):
    avg_last_200 = np.mean(reward_list[0:ep])
    avg_rewards.append(avg_last_200)
    print(f'Episode: {ep}, Average Reward: {avg_last_200}')

# Plot the graph of reward averages
plt.figure(figsize=(14, 6))
plt.plot(range(avg_after_episodes, total_episodes + avg_after_episodes, avg_after_episodes), avg_rewards, marker='o', linestyle='-', color='black')
plt.xlabel('Episodes')
plt.ylabel('Reward Average')
plt.title('Average Reward Over Episodes For Random Actions')
plt.grid(True)
plt.show()

# Show the table of descriptive statistics of average rewards
pd.Series(avg_rewards).describe()