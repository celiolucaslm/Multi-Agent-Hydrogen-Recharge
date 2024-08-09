from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from env.multi_hydrogen_recharge import MultiHydrogenRecharge
import numpy as np
import torch
import matplotlib.pyplot as plt

# Definir os parâmetros padrão para rodar a simulação do ambiente
seed = 42
num_vehicles = 4
num_commands = 4

# -----------------------------------------------------------------------

torch.manual_seed(seed)
np.random.seed(seed)

env = MultiHydrogenRecharge(num_vehicles=num_vehicles, num_commands=num_commands, seed=seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = [env._get_observation().shape for agent in env.vehicles]

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

agent = MADDPG(state_dims=state_dim,
                action_dims=action_dim,
                one_hot=False,
                n_agents=n_agents,
                agent_ids=agent_ids,
                max_action=max_action,
                min_action=min_action,
                discrete_actions=discrete_actions,
                device=device)

# Define os parâmetros de treino do algoritmo
episodes = 10000
max_steps = 10
epsilon = 1
eps_end = 0.01
eps_decay = 0.995
avg_after_episodes = 200


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
        # if any(truncation.values()) or any(termination.values()):
        #     break

    # Save the total episode reward
    score = sum(agent_reward.values())
    agent.scores.append(score)

    # Update epsilon for exploration
    epsilon = max(eps_end, epsilon * eps_decay)

    print('Actual Episode:', ep, '/ Reward:', score, '/ Epsilon:', epsilon)

    # Print average reward of the last 500 episodes
    if ep % avg_after_episodes == 0 and ep != 0:
        avg_last_200 = np.mean(agent.scores)
        print(f'Episode: {ep}, Average Reward: {avg_last_200}')

# ---------------------------------------------------------------------------

# Lista para armazenar as médias das recompensas a cada 200 episódios
avg_rewards = []

# Coleta dos valores de epsilon a cada episódio
eps = []
eps_values = []
for ep in range(episodes):
    epsilon = max(eps_end, epsilon * eps_decay)
    eps_values.append(epsilon)
    eps.append(ep)

# Número total de episódios
total_episodes = len(agent.scores)

# Calculando a média das últimas 200 recompensas a cada 500 episódios
for ep in range(avg_after_episodes, total_episodes+1, avg_after_episodes):
    avg_last_200 = np.mean(agent.scores[0:ep])
    avg_rewards.append(avg_last_200)
    print(f'Episode: {ep}, Average Reward: {avg_last_200}')


# Plota o gráfico das médias de recompensa
plt.figure(figsize=(14, 6))
plt.plot(range(avg_after_episodes, total_episodes + 1, avg_after_episodes), avg_rewards, marker='o', linestyle='-', color='black')
plt.xlabel('Épisodes')
plt.ylabel('Récompense Moyenne')
plt.title("Récompense Moyenne Au Cours Des Épisodes Pour Entraîner L'Algorithme")
plt.grid(True)
plt.show()

# Save the built algorithm
checkpoint_path = "maddpg_agent"
agent.saveCheckpoint(checkpoint_path)

import pandas as pd
pd.Series(avg_rewards).describe()

# Gerar o gráfico
fig, ax1 = plt.subplots(figsize=(14, 6))

# Eixo y para epsilon
ax1.set_xlabel('Épisodes')
ax1.set_ylabel('Epsilon', color='blue')
line1, = ax1.plot(eps, eps_values, color='blue', linestyle='--', label='Epsilon') 
ax1.tick_params(axis='y', labelcolor='blue')

# Eixo y para recompensas médias
ax2 = ax1.twinx()
ax2.set_ylabel('Récompense Moyenne', color='black')
line2, = ax2.plot(range(avg_after_episodes, episodes + 1, avg_after_episodes), avg_rewards, marker='o', linestyle='-', color='black', label='Récompense Moyenne')
ax2.tick_params(axis='y', labelcolor='black')

ax1.legend(loc='upper left', bbox_to_anchor=(0.8, 0.8))
ax2.legend(loc='upper left', bbox_to_anchor=(0.8, 0.75))


fig.tight_layout()
plt.title("Valeur d'Epsilon et Récompense Moyenne Sur Les Épisodes Pour Entraîner L'Algorithme")
plt.grid(True)
plt.show()