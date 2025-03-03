from env.multi_hydrogen_recharge import MultiHydrogenRecharge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set the default parameters for running the environment simulation
seed = 30
num_vehicles = 5

# --------------------------------------------------------------------

np.random.seed(seed)

env = MultiHydrogenRecharge(num_vehicles=num_vehicles, seed=seed)

# Defines the test parameters for the environment's random actions
num_episodes = 20000
max_steps = 20
avg_after_episodes = 200

# Stores the rewards
reward_list = []
num_steps_list = []

# External loop for episodes
for episode in range(num_episodes):
  env.reset()
  vehicle_rewards = {i: 0 for i in range(env.num_vehicles)}

  num_steps = 0

  for step in range(max_steps):
      
    # Vehicles take random action
    #actions = np.random.rand(env.num_vehicles, 4) # Random actions
    actions = np.array([[1., 0., 0., 1.] for _ in range(env.num_vehicles)])

    # Execute the action and take the next observation, reward and done (terminal state)
    observation, rewards, done = env.step(actions)
    
    # After the end of the episode, keep the vehicle rewards
    for i, reward in enumerate(rewards):
        vehicle_rewards[i] += reward

    num_steps += 1
    
    # Stop episode if all agents have terminated
    if all(done.values()):
        break

  score = sum(vehicle_rewards.values())
  reward_list.append(score)
  num_steps_list.append(num_steps)
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

# Calculate the standard deviation of rewards every 200 episodes
std_rewards = np.std(avg_rewards)

# Plot the graph of reward averages with standard deviation
fig, ax1 = plt.subplots(figsize=(14, 6))

color = 'tab:blue'
ax1.set_xlabel('Épisodes')
ax1.set_ylabel('Moyenne des Récompenses', color=color)
ax1.plot(range(avg_after_episodes, total_episodes + 1, avg_after_episodes), avg_rewards, marker='o', linestyle='-', color=color)
ax1.fill_between(range(avg_after_episodes, total_episodes + 1, avg_after_episodes),
                 np.array(avg_rewards) - np.array(std_rewards),
                 np.array(avg_rewards) + np.array(std_rewards),
                 color=color, alpha=0.2)
ax1.tick_params(axis='y', labelcolor=color)

plt.title("Moyenne des Récompenses au Cours des Épisodes pour Actions Aléatoires")
fig.tight_layout()
plt.show()

print('Descriptive statistics of the number of steps')
print(pd.Series(num_steps_list).describe())

# Show the table of descriptive statistics of average rewards
print('Descriptive statistics of average rewards')
print(pd.Series(avg_rewards).describe())