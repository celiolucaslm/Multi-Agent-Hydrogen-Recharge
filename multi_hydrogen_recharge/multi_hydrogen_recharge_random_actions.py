from env.multi_hydrogen_recharge import MultiHydrogenRecharge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Definir os parâmetros padrão para rodar a simulação do ambiente
seed = 42
num_vehicles = 4
num_commands = 4

# --------------------------------------------------------------------

np.random.seed(seed)

env = MultiHydrogenRecharge(num_vehicles=num_vehicles, num_commands=num_commands, seed=seed)

# Define os parâmetros de teste do ambiente
num_episodes = 10000
max_steps = 10
avg_after_episodes = 200

# Armazena as recompensas
reward_list = []

# Boucle externe pour les episodes
for episode in range(num_episodes):
  env.reset()
  vehicle_rewards = {i: 0 for i in range(env.num_vehicles)}

  for step in range(max_steps):
      
      # Vehicule prendre une action aleatoire
      actions = np.random.rand(env.num_vehicles, 3)
      
      # Execute l action et prendre le prochaine observation,  recompense et point d'arret
      observation, rewards, done = env.step(actions)

      # Apres le fin du episode, garder les recompenses des vehicules
      for i, reward in enumerate(rewards):
          vehicle_rewards[i] += reward

  score = sum(vehicle_rewards.values())
  reward_list.append(score)
  print('Actual Episode', episode, '/ Reward: ', score)

  # Print average reward of the last 200 episodes
  if episode % avg_after_episodes == 0 and episode != 0:
        avg_last_200 = np.mean(reward_list)
        print(f'Episode: {episode}, Average Reward: {avg_last_200}')

# Lista para armazenar as médias das recompensas a cada 200 episódios
avg_rewards = []

# Número total de episódios
total_episodes = len(reward_list)

for ep in range(200, total_episodes+1, avg_after_episodes):
    avg_last_200 = np.mean(reward_list[0:ep])
    avg_rewards.append(avg_last_200)
    print(f'Episode: {ep}, Average Reward: {avg_last_200}')

# Plota o gráfico das médias de recompensa
plt.figure(figsize=(14, 6))
plt.plot(range(avg_after_episodes, total_episodes + avg_after_episodes, avg_after_episodes), avg_rewards, marker='o', linestyle='-', color='black')
plt.xlabel('Épisodes')
plt.ylabel('Récompense Moyenne')
plt.title('Récompense Moyenne Au Cours Des Épisodes Pour Des Actions Aléatoires')
plt.grid(True)
plt.show()

# Mostrar a tabela das estatísticas descritivas das recompensas médias
pd.Series(avg_rewards).describe()