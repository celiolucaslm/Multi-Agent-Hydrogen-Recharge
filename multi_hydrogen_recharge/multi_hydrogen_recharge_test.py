from agilerl.algorithms.maddpg import MADDPG
from env.multi_hydrogen_recharge import MultiHydrogenRecharge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Definir os parâmetros padrão para rodar a simulação do ambiente
seed = 42
num_vehicles = 4
num_commands = 4

# ---------------------------------------------------------------------

np.random.seed(seed)

# Load the built algorithm
checkpoint_path = "maddpg_agent"
agent = MADDPG.load(checkpoint_path)

# Armazena as recompensas
agent.scores = []

env = MultiHydrogenRecharge(num_vehicles=num_vehicles, num_commands=num_commands, seed=seed)

# Define os parâmetros de teste do algoritmo
episodes = 10000
max_steps = 10
avg_after_episodes = 200

for ep in range(episodes):
    state = env.reset() # Reset environment at start of episode
    agent_reward = {i: 0 for i in range(env.num_vehicles)}

    for _ in range(max_steps):

        # Get next action from agent
        cont_actions, discrete_action = agent.getAction(
            state,
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

        # Stop episode if any agents have terminated
        # if any(truncation.values()) or any(termination.values()):
        #     break

    # Save the total episode reward
    score = sum(agent_reward.values())
    agent.scores.append(score)

    print('Actual Episode:', ep, '/ Reward:', score)

    # Print average reward of the last 200 episodes
    if ep % avg_after_episodes == 0 and ep != 0:
        avg_last_200 = np.mean(agent.scores)
        print(f'Episode: {ep}, Average Reward: {avg_last_200}')

# ------------------------------------------------------------------------

# Lista para armazenar as médias das recompensas a cada 200 episódios
avg_rewards = []

# Número total de episódios
total_episodes = len(agent.scores)

# Calculando a média das recompensas a cada 200 episódios
for ep in range(200, total_episodes+1, avg_after_episodes):
    avg_last_200 = np.mean(agent.scores[0:ep])
    avg_rewards.append(avg_last_200)
    print(f'Episode: {ep}, Average Reward: {avg_last_200}')


# Plota o gráfico das médias de recompensa
plt.figure(figsize=(14, 6))
plt.plot(range(avg_after_episodes, total_episodes + avg_after_episodes, avg_after_episodes), avg_rewards, marker='o', linestyle='-', color='black')
plt.xlabel('Épisodes')
plt.ylabel('Récompense Moyenne')
plt.title("Récompense Moyenne Sur Les Épisodes Pour Tester L'Algorithme")
plt.grid(True)
plt.show()

# Mostrar a tabela das estatísticas descritivas das recompensas médias
pd.Series(avg_rewards).describe()