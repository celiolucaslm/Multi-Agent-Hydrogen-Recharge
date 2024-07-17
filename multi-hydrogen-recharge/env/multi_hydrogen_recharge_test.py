from agilerl.algorithms.maddpg import MADDPG
from custom_environment import MultiHydrogenRecharge
import numpy as np

checkpoint_path = "env"
agent = MADDPG.load(checkpoint_path)

agent.scores = []

env = MultiHydrogenRecharge(num_vehicles=3, num_commands=3)

episodes = 5000

for ep in range(episodes):
    state = env.reset() # Reset environment at start of episode
    agent_reward = {i: 0 for i in range(env.num_vehicles)}

    for _ in range(15):

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

    # Print average reward of the last 500 episodes
    if ep % 100 == 0:
        if len(agent.scores) >= 100:
            avg_last_500 = np.mean(agent.scores)
        else:
            avg_last_500 = np.mean(agent.scores)
        print(f'Episode: {ep}, Average Reward: {avg_last_500}')


import matplotlib.pyplot as plt

# Lista para armazenar as médias das recompensas a cada 500 episódios
avg_rewards = []

# Número total de episódios
total_episodes = len(agent.scores)

# Calculando a média das últimas 500 recompensas a cada 500 episódios
for ep in range(0, total_episodes, 100):
    if len(agent.scores[ep:ep+100]) >= 100:
        avg_last_500 = np.mean(agent.scores[0:ep+101])
    else:
        avg_last_500 = np.mean(agent.scores)
    avg_rewards.append(avg_last_500)
    print(f'Episode: {ep}, Average Reward: {avg_last_500}')


# Plota o gráfico das médias de recompensa
plt.figure(figsize=(14, 6))
plt.plot(range(0, total_episodes, 100), avg_rewards, marker='o', linestyle='-')
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward Over Episodes')
plt.grid(True)
plt.show()