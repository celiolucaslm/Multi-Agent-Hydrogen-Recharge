from vehicle import Vehicle
from command import Command
from assignments_vehicle import AssignmentsVehicle
import math
import random
from gym import spaces
import numpy as np
from pettingzoo import ParallelEnv

# Declaration MultiHydrogenRecharge (Environment)
class MultiHydrogenRecharge(ParallelEnv):

    metadata = {
        "name": "multi_hydrogen_recharge_v0",
    }

    def __init__(self, num_vehicles, num_commands=None, seed=None):
        super().__init__()

        # Setting the seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Environment settings
        self.num_vehicles = num_vehicles
        self.max_commands = num_vehicles * 4  # Define a maximum number of commands
        self.num_commands = num_commands if num_commands is not None else np.random.poisson(lam=num_vehicles)

        # Defining the observation and action space
        self.observation_space = spaces.Dict({
            'vehicle_position': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'command_positions': spaces.Box(low=0, high=1, shape=(self.max_commands, 2), dtype=np.float32),
            'vehicle_hydrogen': spaces.Box(low=0, high=1, shape=(num_vehicles,), dtype=np.float32),
            'vehicle_remaining_working_time': spaces.Box(low=0, high=1, shape=(num_vehicles,), dtype=np.float32),
            'vehicle_quality_of_service': spaces.Box(low=0, high=1, shape=(num_vehicles,), dtype=np.float32),
            'command_price': spaces.Box(low=0, high=1, shape=(self.max_commands,), dtype=np.float32),
            'command_duration': spaces.Box(low=0, high=1, shape=(self.max_commands,), dtype=np.float32)
        })

        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_vehicles, 3), dtype=np.float32)

        # Initialization of vehicles information
        self.vehicles = [Vehicle(f'V{i+1}', np.random.choice(np.arange(0, 100, 1), size=2), np.random.choice(np.arange(50, 5000, 50)), np.random.choice(np.arange(60, 1440, 10)), np.random.choice(np.arange(1, 5, 1)), np.ones(3) / 3) for i in range(num_vehicles)]

        # Initialization of commands information
        self.commands = [Command(f'C{j+1}', np.random.choice(np.arange(0, 100, 1), size=2), np.random.choice(np.arange(20, 500, 5)), np.random.choice(np.arange(5, 60, 1))) for j in range(self.num_commands)]

        # Matrix of weights of the commands based in different preference of types of vehicle
        commands_weights = []
        
        for _ in range(self.num_commands):
            
            line = [1, 1, 0, 0]

            np.random.shuffle(line)
            
            commands_weights.append(line)

            np.array(commands_weights)
        
        
        for i, commande in enumerate(self.commands):
          commande.weights = commands_weights[i]

        self.match_assignments_vehicule = AssignmentsVehicle(self.commands, self.vehicles)

    def _get_observation(self, vehicle_index):
        vehicle = self.vehicles[vehicle_index]
        vehicle_position = vehicle.position / 100.0  # Assuming positions are within a 100x100 grid
        command_positions = np.zeros((self.max_commands, 2))
        if self.num_commands > 0:
            command_positions[:self.num_commands] = np.array([command.position for command in self.commands]) / 100.0
        vehicle_hydrogen = np.array([vehicle.hydrogen / 5000.0])  # Assuming max hydrogen is 5000
        vehicle_remaining_working_time = np.array([vehicle.remaining_working_time / 1440.0])  # Assuming max working time is 1440
        vehicle_quality_of_service = np.array([vehicle.quality_of_service / 5.0])  # Assuming max quality of service is 5
        command_prices = np.zeros(self.max_commands)
        command_prices[:self.num_commands] = np.array([command.price for command in self.commands]) / 500.0  # Assuming max price is 500
        command_duration = np.zeros(self.max_commands)
        command_duration[:self.num_commands] = np.array([command.duration for command in self.commands]) / 60.0  # Assuming max duration is 60
        num_commands = np.array([self.num_commands / self.max_commands])

        # Calculate distances from this vehicle to each command
        distances = np.zeros(self.max_commands)
        for j, command in enumerate(self.commands):
            distances[j] = calculate_distance(vehicle.position, command.position) / math.sqrt(100**2 + 100**2)  # Normalize distance

        # Get command weights and fill with negative values for non-existing commands
        command_weights = np.full((self.max_commands, 4), -1.0)
        for j, command in enumerate(self.commands):
            command_weights[j] = command.weights

        # Concatenate all observations into a single one-dimensional vector
        observation = np.concatenate([
            vehicle_position,
            distances,
            num_commands,
            vehicle_hydrogen,
            vehicle_remaining_working_time,
            vehicle_quality_of_service,
            command_prices,
            command_duration,
            # command_weights.flatten()  # Flatten the command weights array if needed
        ])

        return observation


    def reset(self):
        self.num_commands = np.random.poisson(lam=self.num_vehicles)
        # Reset of commands information
        self.commands = [Command(f'C{j+1}', np.random.choice(np.arange(0, 100, 1), size=2), np.random.choice(np.arange(20, 500, 5)), np.random.choice(np.arange(5, 60, 1))) for j in range(self.num_commands)]

        # Reset of vehicles information
        self.vehicles = [Vehicle(f'V{i+1}', np.random.choice(np.arange(0, 100, 1), size=2), np.random.choice(np.arange(50, 5000, 50)), np.random.choice(np.arange(60, 1440, 10)), np.random.choice(np.arange(1, 5, 1)), np.ones(3) / 3) for i in range(self.num_vehicles)]
   
        # Matrix of weights of the commands based in different preference of types of vehicle
        commands_weights = []
        
        for _ in range(self.num_commands):
            
            line = [1, 1, 0, 0]

            np.random.shuffle(line)
            
            commands_weights.append(line)

            np.array(commands_weights)
        
        for i, command in enumerate(self.commands):
          command.weights = commands_weights[i]

        self.match_assignments_vehicule.reset()

        # Reset preferences (vehicles and commands)
        for vehicle in self.vehicles:
            vehicle.preference = []
        for command in self.commands:
            command.preference = []

        observations = {}
        for i in range(len(self.vehicles)):
            observations[i] = self._get_observation(i)

        return observations

    def step(self, actions):
        # Update vehicle weights according to action taken
        for i, vehicle in enumerate(self.vehicles):
            vehicle.weights = actions[i]

        # Updates the preference of each vehicle and order with name and Score
        for vehicule in self.vehicles:
            for commande in self.commands:
                score = calculate_vehicle_score(commande, vehicule.weights, vehicule.position)
                vehicule.preference.append((commande.name, score))

        for commande in self.commands:
            for vehicule in self.vehicles:
                score = calculate_command_score(vehicule, commande.weights, commande.position)
                commande.preference.append((vehicule.name, score))

        # Rank the preference of each vehicle and order according to score
        for vehicule in self.vehicles:
            vehicule.preference.sort(key=lambda x: x[1], reverse=True)

        for commande in self.commands:
            commande.preference.sort(key=lambda x: x[1], reverse=True)

        self.match_assignments_vehicule = AssignmentsVehicle(list(self.commands), list(self.vehicles))
        assignments_vehicule = self.match_assignments_vehicule.match()

        for c in self.commands:
            print(f'Preference of the Command {c.name}:', c.preference)
        
        for v in self.vehicles:
            print(f'Preference of the Vehicle {v.name}:', v.preference)   

        print('Assignments:', assignments_vehicule)

        # for v in self.vehicles:
        #     distances = [calculate_distance(v.position, c.position) for c in self.commands]
        #     print(f'Vehicle {v.name}: Distances to commands - {distances}, Hydrogen - {v.hydrogen}, Remaining Working Time - {v.remaining_working_time}, Quality of Service - {v.quality_of_service}')

        # Reward calculation for each vehicle
        rewards = []
        for i, vehicule in enumerate(self.vehicles):
            # Check if the vehicle has a preference
            if vehicule.preference:
                # Iterate over the list of preferences to find the position of the current reward
                current_preference_position = None
                for j, (cmd_name, score) in enumerate(vehicule.preference):
                    if vehicule.job is not None and cmd_name == vehicule.job.name:
                        current_preference_position = j
                        break

                # Assigns the reward based on the position in the preferences list
                if current_preference_position == 0:
                    rewards.append(20)  # If it's the favorite
                elif current_preference_position == 1:
                    rewards.append(10)   # If it's the second most preferred
                elif current_preference_position == 2:
                    rewards.append(0)   # If it's the third most preferred
                else:
                    rewards.append(-40)  # Other cases (less preferred)
            else:
                rewards.append(0)  # If there is no preference

        # Vehicles pick up the command position
        for vehicule in self.vehicles:
            if vehicule.job is not None:
                vehicule.position = vehicule.job.position

        # Vehicles lose hydrogen after a service
        for vehicule in self.vehicles:
            #print(f'Vehicle {vehicule.name}: Vehicle Job - {vehicule.job}')
            if vehicule.job is not None:
                vehicule.hydrogen = vehicule.hydrogen - (vehicule.job.duration * 80)
                # Ensure hydrogen does not go below zero
                #vehicule.hydrogen = max(vehicule.hydrogen, 0.0)


        self.num_commands = np.random.poisson(lam=self.num_vehicles)

        # Reset of commands information
        self.commands = [Command(f'C{j+1}', np.random.choice(np.arange(0, 100, 1), size=2), np.random.choice(np.arange(20, 500, 5)), np.random.choice(np.arange(5, 60, 1))) for j in range(self.num_commands)]

        # Matrix of weights of the commands based in different preference of types of vehicle
        # commands_weights = []
        
        # for _ in range(self.num_commands):
            
        #     line = [1, 1, 0, 0]

        #     np.random.shuffle(line)
            
        #     commands_weights.append(line)

        #     np.array(commands_weights)
        
        # random.shuffle(commands_weights)

        # for i, command in enumerate(self.commands):
        #   command.weights = commands_weights[i]

        # Track the number of steps each vehicle has received a reward <= 0
        if not hasattr(self, 'negative_reward_steps'):
            self.negative_reward_steps = {i: 0 for i in range(self.num_vehicles)}

        for i, reward in enumerate(rewards):
            if reward <= 0:
                self.negative_reward_steps[i] += 1
            else:
                self.negative_reward_steps[i] = 0

        done = {i: self.negative_reward_steps[i] >= 3 for i in range(self.num_vehicles)}

        # Reset preferences (vehicles and commands)
        for vehicle in self.vehicles:
            vehicle.preference = []
        for command in self.commands:
            command.preference = []

        self.match_assignments_vehicule.reset()

        observations = {}
        for i in range(len(self.vehicles)):
            observations[i] = self._get_observation(i)

        # Return the current observation, reward, breakpoint and additional information
        return observations, rewards, done
    
def calculate_distance(position1, position2):
    return math.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2)

def calculate_vehicle_score(command, weights, position):
    max_price = 500.0 
    max_distance = math.sqrt(100**2 + 100**2)  # Assuming positions are within a 100x100 grid
    max_duration = 60.0  # Assuming duration is between 0 and 1

    normalized_price = command.price / max_price
    normalized_distance = calculate_distance(command.position, position) / max_distance
    normalized_duration = command.duration / max_duration

    score = ((normalized_price * weights[0])) - (normalized_distance * weights[1]) - (normalized_duration * weights[2])
    return score

def calculate_command_score(vehicle, weights, position):
    # Normalize each attribute
    max_hydrogen = 5000.0  # Assuming hydrogen is between 0 and 1
    max_distance = math.sqrt(100**2 + 100**2)  # Assuming positions are within a 100x100 grid
    max_working_time = 1440.0  # Assuming working time is between 0 and 1
    max_quality_of_service = 5.0  # Assuming quality of service is between 0 and 1

    normalized_hydrogen = vehicle.hydrogen / max_hydrogen
    normalized_distance = calculate_distance(vehicle.position, position) / max_distance
    normalized_working_time = vehicle.remaining_working_time / max_working_time
    normalized_quality_of_service = vehicle.quality_of_service / max_quality_of_service

    score = ((normalized_hydrogen * weights[0]) - (normalized_distance * weights[1]) + (normalized_working_time * weights[2]) + (normalized_quality_of_service * weights[3]))
    return score