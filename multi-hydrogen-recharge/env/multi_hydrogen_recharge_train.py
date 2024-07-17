from agilerl.algorithms.maddpg import MADDPG
from multi_hydrogen_recharge import MultiHydrogenRecharge
import numpy as np


env = MultiHydrogenRecharge(num_vehicles=3, num_commands=3)

import torch
from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = [env._get_observation().shape for agent in env.vehicles]

action_dim = [env.action_space.shape[0] for agent in env.vehicles]
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
                device=device,
                gamma=0.99)

episodes = 5000
max_steps = 15
epsilon = 1.0
eps_end = 0.01
eps_decay = 0.995


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
        )  # Act in environment
    	
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
        avg_last_500 = np.mean(agent.scores[ep:])
    avg_rewards.append(avg_last_500)
    print(f'Episode: {ep}, Average Reward: {avg_last_500}')


# Plota o gráfico das médias de recompensa
plt.figure(figsize=(14, 6))
plt.plot(range(100, total_episodes + 100, 100), avg_rewards, marker='o', linestyle='-')
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward Over Episodes')
plt.grid(True)
plt.show()

checkpoint_path = "env"
agent.saveCheckpoint(checkpoint_path)


## RETORNAR APENAS O PESOS DADOS PELAS COMANDAS