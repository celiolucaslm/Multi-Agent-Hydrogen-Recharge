import random
import math
import gym
from gym import spaces
import numpy as np
from pettingzoo import ParallelEnv
import matplotlib.pyplot as plt

# Déclaration Class Véhicule

class Vehicle:
    def __init__(self, name, position, hydrogen, indicateur, qualite, weights):
        self.name = name
        self.position = position
        self.hydrogen = hydrogen
        self.indicateur = indicateur
        self.qualite = qualite
        self.weights = weights
        self.preference = []
        self.is_matched = False
        self.job = None
        self.index = 0
        self.reward = 0

    def propose(self):
        commande = self.preference[self.index]
        self.index += 1
        return commande

    def is_available(self):
        return not self.is_matched

    def update_reward(self, score):
        self.reward = score

    def reset_index(self):
        self.index = 0

# Déclaration Class Véhicule

class Command:
    def __init__(self, name, position, prix, duration):
        self.name = name
        self.position = position
        self.prix = prix
        self.duration = duration
        self.weights = [0.25, 0.25, 0.25, 0.25]
        self.preference = []
        self.is_matched = False
        self.vehicle = None
        self.index = 0
        self.reward = 0

    def is_available(self):
        return not self.is_matched

    def propose(self):
        vehicule = self.preference[self.index]
        self.index += 1
        return vehicule

    def update_reward(self, score):
        self.reward = score

    def reset_index(self):
        self.index = 0


# Déclaration Class Assignments (Gale-Shapley)

class Assignments_Vehicule:
    def __init__(self, commandes, vehicules):
        self.assignments = {}
        self.commandes = commandes
        self.vehicules = vehicules
        self.assignment_count = 0

        for commande in commandes:
            self.assignments[commande.name] = commande

        for vehicule in vehicules:
            self.assignments[vehicule.name] = vehicule

    def assign(self, vehicule_name, commande_name):
        commande = self.assignments[commande_name]
        vehicule = self.assignments[vehicule_name]

        commande.is_matched = True
        commande.vehicule = vehicule

        vehicule.is_matched = True
        vehicule.job = commande

        self.assignment_count += 1

    def unassign(self, vehicule_name, commande_name):
        commande = self.assignments[commande_name]
        vehicule = self.assignments[vehicule_name]

        commande.is_matched = False
        commande.vehicule = None

        vehicule.is_matched = False
        vehicule.job = None

        self.assignment_count -= 1

    def reset(self):
        for commande in self.commandes:
            commande.is_matched = False
            commande.vehicule = None
            commande.reward = 0
            commande.reset_index()

        for vehicule in self.vehicules:
            vehicule.is_matched = False
            vehicule.job = None
            vehicule.reward = 0
            vehicule.reset_index()

    def match(self):
        proposals = {commande.name: [] for commande in self.commandes}

        # Loop until all vehicles are matched
        while True:
            # Find all unmatched vehicles
            unmatched_vehicles = [vehicule for vehicule in self.vehicules if not vehicule.is_matched]
            if not unmatched_vehicles:
                break

            for vehicule in unmatched_vehicles:
                # Vehicule makes a proposal to the next commande in its preference list
                commande_name, score = vehicule.propose()
                commande = self.assignments[commande_name]
                commande.update_reward(score)

                proposals[commande_name].append((vehicule, score))

            # Process proposals for each commande
            for commande_name, proposers in proposals.items():
              if proposers:
                  proposers.sort(key=lambda x: x[1], reverse=True)  # Sort proposers by score
                  best_proposer, best_score = proposers[0]
                  commande = self.assignments[commande_name]

                  if commande.is_available():
                      self.assign(best_proposer.name, commande_name)
                  else:
                      current_vehicule = commande.vehicule
                      current_vehicule_score = next((score for v, score in proposers if v.name == current_vehicule.name), None)

                      # Verifica se current_vehicule_score é diferente de None antes de fazer a comparação
                      if current_vehicule_score is not None and best_score > current_vehicule_score:
                          self.unassign(current_vehicule.name, commande_name)
                          self.assign(best_proposer.name, commande_name)


            # Clear proposals after processing
            proposals = {commande.name: [] for commande in self.commandes}

        return self.sets()

    def sets(self):
        matches = {}
        for i in self.assignments:
            assignment = self.assignments[i]
            if isinstance(assignment, Vehicle) and assignment.is_matched:
                matches[frozenset([assignment.name, assignment.job.name])] = True
        return list(matches.keys())


def calculate_distance(position1, position2):
    return math.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2)

def calculate_score_vehicule(objet, poids, position):
    score = (objet.prix * poids[0]) - (calculate_distance(objet.position, position) * poids[1]) - (objet.duration * poids[2])
    return score

def calculate_score_commande(objet, poids, position):
    score = ((objet.hydrogen * poids[0]) - (calculate_distance(objet.position, position) * poids[1]) + (objet.indicateur * poids[2]) + (objet.qualite * poids[3]))
    return score

class MultiHydrogenRecharge(ParallelEnv):

    metadata = {
        "name": "multi_hydrogen_recharge_v0",
    }

    def __init__(self, num_vehicles=3, num_commands=3):
        super().__init__()

        # Paramètres de l'environnement
        self.num_vehicles = num_vehicles
        self.num_commands = num_commands

        # Définition de l'espace de observation et action
        self.observation_space = spaces.Dict({
            'vehicle_positions': spaces.Box(low=0, high=1, shape=(num_vehicles, 2), dtype=np.float32),
            'command_positions': spaces.Box(low=0, high=1, shape=(num_commands, 2), dtype=np.float32),
            'vehicle_hydrogen': spaces.Box(low=0, high=1, shape=(num_vehicles,), dtype=np.float32),
            'command_price': spaces.Box(low=0, high=1, shape=(num_commands,), dtype=np.float32),
            'command_duration': spaces.Box(low=0, high=1, shape=(num_commands,), dtype=np.float32)
        })
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_vehicles, 3))

        # self.position = position
        # self.hydrogen = hydrogen
        # self.indicateur = indicateur
        # self.qualite = qualite
        # self.weights = weights

        hydrogen_values = [0.2, 0.2, 0.9]
        indicateur_values = [0.2, 0.9, 0.2]
        qualite_values = [0.9, 0.2, 0.2]

        # Initialisation des informations des véhicules
        self.vehicles = [Vehicle(f'V{i+1}', np.random.choice(np.arange(0, 1.1, 0.1), size=2), hydrogen_values[i], indicateur_values[i], qualite_values[i], np.ones(3) / 3) for i in range(num_vehicles)]

        # self.position = position
        # self.prix = prix
        # self.duration = duration

        # Initialisation des informations des commandes
        prix = [0.7, 0.8, 0.9]
        duration = [0.9, 0.8, 0.7]
        self.commands = [Command(f'C{j+1}', np.random.choice(np.arange(0, 1.1, 0.1), size=2), np.random.choice(np.arange(0.5, 1.1, 0.5)), np.random.choice(np.arange(0.5, 1.1, 0.5))) for j in range(num_commands)]

        soma_x = 0
        soma_y = 0

        for vehicule in self.vehicles:
          soma_x += vehicule.position[0]
          soma_y += vehicule.position[1]

        media = [soma_x/self.num_vehicles, soma_y/self.num_vehicles]

        commands_weights = [[1, 1, 0.01, 0.01],
                            [0.01, 1, 1, 0.01],
                            [0.01, 1, 0.01, 1]]
        
        random.shuffle(commands_weights)
        
        for i, commande in enumerate(self.commands):
          commande.weights = commands_weights[i]

        # for commande in self.commands:
        #   if (calculate_distance(commande.position, media) > 0.4):
        #     commande.weights = [0.1, 0.1, 0.1, 0.7]
        #   else:
        #     commande.weigths = [0.7, 0.1, 0.1, 0.1]

        self.match_assignments_vehicule = Assignments_Vehicule(self.commands, self.vehicles)


    def _get_observation(self):
        # Retour de l'observation initiale
        # Coletar todas as observações em arrays separados
        vehicle_positions = np.array([vehicle.position for vehicle in self.vehicles]).flatten()
        command_positions = np.array([command.position for command in self.commands]).flatten()
        vehicle_hydrogen = np.array([vehicle.hydrogen for vehicle in self.vehicles])
        vehicle_indicateur = np.array([vehicle.indicateur for vehicle in self.vehicles])
        vehicle_qualite = np.array([vehicle.qualite for vehicle in self.vehicles])
        command_prices = np.array([command.prix for command in self.commands])
        command_duration = np.array([command.duration for command in self.commands])
        #vehicle_weights = np.array([vehicle.weights for vehicle in self.vehicles]).flatten()
        command_weights = np.array([command.weights for command in self.commands]).flatten()

        # Concatenar todas as observações em um único vetor unidimensional
        observation = np.concatenate([
            vehicle_positions,
            command_positions,
            # vehicle_hydrogen,
            # vehicle_indicateur,
            # vehicle_qualite,
            command_prices,
            command_duration,
            #vehicle_weights,
            command_weights
        ])

        return observation

    def reset(self):
        self.commands = [Command(f'C{j+1}', np.random.choice(np.arange(0, 1.1, 0.1), size=2), np.random.choice(np.arange(0.5, 1.1, 0.5)), np.random.choice(np.arange(0.5, 1.1, 0.5))) for j in range(self.num_commands)]
        soma_x = 0
        soma_y = 0

        hydrogen_values = [0.2, 0.2, 0.9]
        indicateur_values = [0.2, 0.9, 0.2]
        qualite_values = [0.9, 0.2, 0.2]

        # Initialisation des informations des véhicules
        self.vehicles = [Vehicle(f'V{i+1}', np.random.choice(np.arange(0, 1.1, 0.1), size=2), hydrogen_values[i], indicateur_values[i], qualite_values[i], np.ones(3) / 3) for i in range(self.num_vehicles)]

        for vehicule in self.vehicles:
          soma_x += vehicule.position[0]
          soma_y += vehicule.position[1]

        media = [soma_x/self.num_vehicles, soma_y/self.num_vehicles]

        commands_weights = [[1, 1, 0.01, 0.01],
                            [0.01, 1, 1, 0.01],
                            [0.01, 1, 0.01, 1]]

        random.shuffle(commands_weights)
        
        for i, commande in enumerate(self.commands):
          commande.weights = commands_weights[i]

        # for commande in self.commands:
        #   if (calculate_distance(commande.position, media) > 0.4):
        #     commande.weights = [0.1, 0.1, 0.2, 0.6]
        #   else:
        #     commande.weigths = [0.6, 0.2, 0.1, 0.1]


        self.match_assignments_vehicule.reset()
        # Reset des préferences (véhicules et commandes)
        for vehicule in self.vehicles:
            vehicule.preference = []
        for commande in self.commands:
            commande.preference = []

        # Colete todas as observações em arrays separados
        vehicle_positions = np.array([vehicle.position for vehicle in self.vehicles]).flatten()  # 6 elementos
        command_positions = np.array([command.position for command in self.commands]).flatten()  # 6 elementos
        vehicle_hydrogen = np.array([vehicle.hydrogen for vehicle in self.vehicles])  # 3 elementos
        vehicle_indicateur = np.array([vehicle.indicateur for vehicle in self.vehicles])  # 3 elementos
        vehicle_qualite = np.array([vehicle.qualite for vehicle in self.vehicles])  # 3 elementos
        command_prices = np.array([command.prix for command in self.commands])  # 3 elementos
        command_duration = np.array([command.duration for command in self.commands])  # 3 elementos
        command_weights = np.array([command.weights for command in self.commands]).flatten()  # 6 elementos

        # Construa a observação completa para cada veículo
        full_observation = np.concatenate([
            vehicle_positions,
            command_positions,
            # vehicle_hydrogen,
            # vehicle_indicateur,
            # vehicle_qualite,
            command_prices,
            command_duration,
            command_weights
        ])

        observations = {}
        for i in range(len(self.vehicles)):
            observations[i] = full_observation

        # Retour d'observation initiale (concatène les informations des véhicules et commandes)         
        return observations

    def step(self, actions):

        # Mise à jour les poids des véhicules en fonction de l'action prise
        for vehicle in self.vehicles:
            if vehicle in actions:
                vehicle.weights = actions[vehicle]

        # Actualise la préference de chaque véhicule et commande avec le nom et le Score
        for vehicule in self.vehicles:
            for commande in self.commands:
                score = calculate_score_vehicule(commande, vehicule.weights, vehicule.position)
                vehicule.preference.append((commande.name, score))

        for commande in self.commands:
            for vehicule in self.vehicles:
                score = calculate_score_commande(vehicule, commande.weights, commande.position)
                commande.preference.append((vehicule.name, score))

        # Classez la préférence de chaque véhicule et commande en fonction du score
        for vehicule in self.vehicles:
            vehicule.preference.sort(key=lambda x: x[1], reverse=True)

        for commande in self.commands:
            commande.preference.sort(key=lambda x: x[1], reverse=True)

        self.match_assignments_vehicule = Assignments_Vehicule(self.commands, self.vehicles)
        assignments_vehicule = self.match_assignments_vehicule.match()

        # Calcul de la récompense pour chaque véhicule
        rewards = []
        for i, vehicule in enumerate(self.vehicles):
            # Verifica se o veículo tem uma preferência
            if vehicule.preference:
                # Obtém a recompensa do par mais preferido na lista de preferências
                max_preference = vehicule.preference[0][1]

                # Itera sobre a lista de preferências para encontrar a posição da recompensa atual
                current_preference_position = None
                for j, (cmd_name, score) in enumerate(vehicule.preference):
                    if cmd_name == vehicule.job.name:
                        current_preference_position = j
                        break

                # Atribui a recompensa com base na posição na lista de preferências
                if current_preference_position == 0:
                    rewards.append(10)  # Se for o mais preferido
                elif current_preference_position == 1:
                    rewards.append(0)   # Se for o segundo mais preferido
                else:
                    #current_preference_position == 2:
                    rewards.append(-10)   # Se for o terceiro mais preferido
            #     else:
            #         rewards.append(-50)    # Outros casos (menos preferidos)
            # else:
            #     rewards.append(0)  # Caso não haja preferências

        # # Vehicules pegam a posição da comanda
        # for vehicule in self.vehicles:
        #     if vehicule.job:
        #         vehicule.position = vehicule.job.position

        # print("Préférence des véhicules:")
        # for vehicule in self.vehicles:
        #     print(f"{vehicule.name}:")
        #     for commande_name, score in vehicule.preference:
        #        print(f"\tCommande: {commande_name} - Score: {score}")

        # print("\nPréférence des commandes:")
        # for commande in self.commands:
        #     print(f"{commande.name}:")
        #     for vehicule_name, score in commande.preference:
        #         print(f"\tVéhicule: {vehicule_name} - Score: {score}")


        # print("\n\nAffectation Vehicule proposant")
        # self.match_assignments_vehicule = Assignments_Vehicule(self.commands, self.vehicles)
        # assignments_vehicule = self.match_assignments_vehicule.match()
        # print("Matches:", assignments_vehicule)

        # Reset pour nouvelles commandes
        #self.reset()

        done = {0: False, 1: False, 2: False}

        # Colete todas as observações em arrays separados
        vehicle_positions = np.array([vehicle.position for vehicle in self.vehicles]).flatten()  # 6 elementos
        command_positions = np.array([command.position for command in self.commands]).flatten()  # 6 elementos
        vehicle_hydrogen = np.array([vehicle.hydrogen for vehicle in self.vehicles])  # 3 elementos
        vehicle_indicateur = np.array([vehicle.indicateur for vehicle in self.vehicles])  # 3 elementos
        vehicle_qualite = np.array([vehicle.qualite for vehicle in self.vehicles])  # 3 elementos
        command_prices = np.array([command.prix for command in self.commands])  # 3 elementos
        command_duration = np.array([command.duration for command in self.commands])  # 3 elementos
        command_weights = np.array([command.weights for command in self.commands]).flatten()  # 6 elementos

        # Construa a observação completa para cada veículo
        full_observation = np.concatenate([
            vehicle_positions,
            command_positions,
            # vehicle_hydrogen,
            # vehicle_indicateur,
            # vehicle_qualite,
            command_prices,
            command_duration,
            command_weights
        ])

        observations = {}
        for i in range(len(self.vehicles)):
            observations[i] = full_observation

        # Retorne a observação atual, a recompensa, o ponto de parada e as informações adicionais
        return observations, rewards, done

env = MultiHydrogenRecharge(num_vehicles=3, num_commands=3)

# Definit le nombre d'episodes
num_episodes = 5000

score_list = []

# Boucle externe pour les episodes
for episode in range(num_episodes):
  #print(f"Episode {episode+1}")
  env.reset()
  vehicle_rewards = {i: 0 for i in range(env.num_vehicles)}

  for step in range(15):
      # print("\nStep", step+1)

      # Vehicule prendre une action aleatoire
      actions = np.random.rand(3, 3)
      # print("Action des vehicules:", actions)

      # Execute l action et prendre le prochaine observation,  recompense et point d'arret
      observation, rewards, done = env.step(actions)
      #print("Nouvelle observation:", observation)
      # print("Recompense des véhicules:", rewards)

      # Apres le fin du episode, garder les recompenses des vehicules
      for i, reward in enumerate(rewards):
          vehicle_rewards[i] += reward

  #vehicle_rewards = np.array(vehicle_rewards)
  # print(vehicle_rewards)
  score = sum(vehicle_rewards.values())
  score_list.append(score)
  print('Actual Episode', episode, '/ Reward: ', score)

    # Print average reward of the last 500 episodes
if episode % 100 == 0:
    if len(score_list) >= 100:
        avg_last_500 = np.mean(score_list)
    else:
        avg_last_500 = np.mean(score_list)
    print(f'Episode: {episode}, Average Reward: {avg_last_500}')

# print("\n-----------------MOYENNE-------------------")
# print('MEAN SCORE:', np.array(score_list).mean())
#for i in range(env.num_vehicles):
#  print(f"\nMoyenne V{i+1}: {vehicle_rewards[i].mean()}")

import matplotlib.pyplot as plt

# Lista para armazenar as médias das recompensas a cada 500 episódios
avg_rewards = []

# Número total de episódios
total_episodes = len(score_list)

# Calculando a média das últimas 500 recompensas a cada 500 episódios
for ep in range(0, total_episodes, 100):
    if len(score_list[ep:ep+100]) >= 100:
        avg_last_500 = np.mean(score_list[0:ep+101])
    else:
        avg_last_500 = np.mean(score_list[ep:])
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