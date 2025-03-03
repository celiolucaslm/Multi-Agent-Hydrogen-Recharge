from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from env.multi_hydrogen_recharge import MultiHydrogenRecharge
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import time

# Set the default parameters for running the environment simulation
seed = 42
num_vehicles = 5

# -----------------------------------------------------------------------

torch.manual_seed(seed)
np.random.seed(seed)

env = MultiHydrogenRecharge(num_vehicles=num_vehicles, seed=seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = [env._get_observation(vehicle_index=i).shape for i, agent in enumerate(env.vehicles)]

action_dim = [env.action_space.shape[1] for agent in env.vehicles]
discrete_actions = False
max_action = [env.action_space.high[0] for agent in env.vehicles]
min_action = [env.action_space.low[0] for agent in env.vehicles]

n_agents = env.num_vehicles
agent_ids = [i for i in range(env.num_vehicles)]
field_names = ["state", "action", "reward", "next_state", "done"]

memory = MultiAgentReplayBuffer(memory_size=1_000_000,
                                field_names=field_names,
                                agent_ids=agent_ids,
                                device=device)

NET_CONFIG = {
      "arch": "mlp",  # Network architecture
      "h_size": [256, 256],  # Hidden size
  }

agent = MADDPG(state_dims=state_dim,
                action_dims=action_dim,
                one_hot=False,
                n_agents=n_agents,
                agent_ids=agent_ids,
                max_action=max_action,
                min_action=min_action,
                discrete_actions=discrete_actions,
                device=device,
                learn_step=5,
                batch_size=512,
                net_config=NET_CONFIG)

# Define the algorithm's training parameters
episodes = 10000
max_steps = 20
epsilon = 1.0
eps_end = 0.01
eps_decay = 0.995
avg_after_episodes = 100

start_time = time.time()

for ep in range(episodes):
    state = env.reset() # Reset environment at start of episode
    agent_reward = {i: 0 for i in range(env.num_vehicles)}

    for _ in range(max_steps):

        # Get next action from agent
        cont_actions, discrete_action = agent.getAction(
            state, epsilon,
        )
        if agent.discrete_actions:
            action = discrete_action
        else:
            action = cont_actions

        next_state, reward, done = env.step(
            action
        )  # Act in the environment
    	
        # Save experiences to replay buffer
        memory.save2memory(state, cont_actions, reward, next_state, done)

        for i, r in enumerate(reward):
            agent_reward[i] += r

        # Learn according to learning frequency
        if (memory.counter % agent.learn_step == 0) and (len(
                memory) >= agent.batch_size):
            experiences = memory.sample(agent.batch_size) # Sample replay buffer
            
            agent.learn(experiences) # Learn according to agent's RL algorithm

        # Update the state
        state = next_state

        # Stop episode if any agents have terminated
        if all(done.values()):
            break

    # Save the total episode reward
    score = sum(agent_reward.values())
    agent.scores.append(score)

    # Update epsilon for exploration
    epsilon = max(eps_end, epsilon * eps_decay)

    print('Actual Episode:', ep, '/ Reward:', score, '/ Epsilon:', epsilon)

    # Print average reward of the last 500 episodes
    if ep % avg_after_episodes == 0 and ep != 0:
        avg_last_100 = np.mean(agent.scores)
        print(f'Episode: {ep}, Average Reward: {avg_last_100}')

end_time = time.time()

print(f'Training Time: {(end_time - start_time) / 60} minutes')

# ---------------------------------------------------------------------------

# List to store the average rewards every 100 episodes
avg_rewards = []

# Collecting epsilon values every episode
eps = []
eps_values = []
epsilon = 1.0
for ep in range(episodes):
    epsilon = max(eps_end, epsilon * eps_decay)
    eps_values.append(epsilon)
    eps.append(ep)

# Total number of episodes
total_episodes = len(agent.scores)

# Calculating average rewards every 100 episodes
for ep in range(avg_after_episodes, total_episodes+1, avg_after_episodes):
    avg_last_100 = np.mean(agent.scores[0:ep])
    avg_rewards.append(avg_last_100)
    print(f'Episode: {ep}, Average Reward: {avg_last_100}')

# Plot the graph of reward averages
plt.figure(figsize=(14, 6))
plt.plot(range(avg_after_episodes, total_episodes + 1, avg_after_episodes), avg_rewards, marker='o', linestyle='-', color='black')
plt.xlabel('Épisodes')
plt.ylabel('Récompense Moyenne')
plt.title("Récompense Moyenne au Cours des Épisodes pour Entraîner l'Algorithme")
plt.grid(True)
plt.show()

# Generate the graph to compare epsilon and average rewards
fig, ax1 = plt.subplots(figsize=(14, 6))

# Y-axis for epsilon
ax1.set_xlabel('Épisodes')
ax1.set_ylabel('Epsilon', color='blue')
line1, = ax1.plot(eps, eps_values, color='blue', linestyle='--', label='Epsilon') 
ax1.tick_params(axis='y', labelcolor='blue')

# Y-axis for average rewards
ax2 = ax1.twinx()
ax2.set_ylabel('Récompense Moyenne', color='black')
line2, = ax2.plot(range(avg_after_episodes, episodes + 1, avg_after_episodes), avg_rewards, marker='o', linestyle='-', color='black', label='Récompense Moyenne')
ax2.tick_params(axis='y', labelcolor='black')

ax1.legend(loc='upper left', bbox_to_anchor=(0.8, 0.85))
ax2.legend(loc='upper left', bbox_to_anchor=(0.8, 0.8))

fig.tight_layout()
plt.title("Valeur d'Epsilon et Récompense Moyenne au Cours des Épisodes pour Entraîner l'Algorithme")
plt.grid(True)
plt.show()

# Show the table of descriptive statistics of average rewards
print(pd.Series(avg_rewards).describe())

# Save the built algorithm
checkpoint_path = "maddpg_agent_5v"
agent.saveCheckpoint(checkpoint_path)