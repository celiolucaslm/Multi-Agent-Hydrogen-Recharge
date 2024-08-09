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

    def __init__(self, num_vehicles, num_commands, seed=None):
        super().__init__()

        # Setting the seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)


        # Environment settings
        self.num_vehicles = num_vehicles
        self.num_commands = num_commands

        # Defining the observation and action space
        self.observation_space = spaces.Dict({
            'vehicle_positions': spaces.Box(low=0, high=1, shape=(num_vehicles, 2), dtype=np.float32),
            'command_positions': spaces.Box(low=0, high=1, shape=(num_commands, 2), dtype=np.float32),
            'vehicle_hydrogen': spaces.Box(low=0, high=1, shape=(num_vehicles,), dtype=np.float32),
            'vehicle_remaining_working_time': spaces.Box(low=0, high=1, shape=(num_vehicles,), dtype=np.float32),
            'vehicle_quality_of_service': spaces.Box(low=0, high=1, shape=(num_vehicles,), dtype=np.float32),
            'command_price': spaces.Box(low=0, high=1, shape=(num_commands,), dtype=np.float32),
            'command_duration': spaces.Box(low=0, high=1, shape=(num_commands,), dtype=np.float32)
        })

        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_vehicles, 3), dtype=np.float32)

        # Initialization of vehicles information
        self.vehicles = [Vehicle(f'V{i+1}', np.random.choice(np.arange(0, 1.1, 0.1), size=2), np.random.choice(np.arange(0.1, 1.1, 0.2)), np.random.choice(np.arange(0.1, 1.1, 0.1)), np.random.choice(np.arange(0.2, 1.1, 0.4)), np.ones(3) / 3) for i in range(num_vehicles)]

        # Initialization of commands information
        self.commands = [Command(f'C{j+1}', np.random.choice(np.arange(0, 1.1, 0.1), size=2), np.random.choice(np.arange(0.2, 1.1, 0.1)), np.random.choice(np.arange(0.1, 1.1, 0.1))) for j in range(num_commands)]

        # Matrix of weights of the commands based in different preference of types of vechicle
        commands_weights = []
        
        for _ in range(self.num_commands):
            
            line = [1, 1, 0, 0]

            np.random.shuffle(line)
            
            commands_weights.append(line)

            np.array(commands_weights)
        
        
        for i, commande in enumerate(self.commands):
          commande.weights = commands_weights[i]

        self.match_assignments_vehicule = AssignmentsVehicle(self.commands, self.vehicles)


    def _get_observation(self):
        # Return from initial observation
        vehicle_positions = np.array([vehicle.position for vehicle in self.vehicles]).flatten()
        command_positions = np.array([command.position for command in self.commands]).flatten()
        vehicle_hydrogen = np.array([vehicle.hydrogen for vehicle in self.vehicles])
        vehicle_remaining_working_time = np.array([vehicle.remaining_working_time for vehicle in self.vehicles])
        vehicle_quality_of_service = np.array([vehicle.quality_of_service for vehicle in self.vehicles])
        command_prices = np.array([command.price for command in self.commands])
        command_duration = np.array([command.duration for command in self.commands])
        command_weights = np.array([command.weights for command in self.commands]).flatten()

        # Concatenate all observations into a single one-dimensional vector
        observation = np.concatenate([
            vehicle_positions,
            command_positions,
            vehicle_hydrogen,
            vehicle_remaining_working_time,
            vehicle_quality_of_service,
            command_prices,
            command_duration,
            command_weights
        ])

        return observation

    def reset(self):
        # Reset of commands information
        self.commands = [Command(f'C{j+1}', np.random.choice(np.arange(0, 1.1, 0.1), size=2), np.random.choice(np.arange(0.2, 1.1, 0.1)), np.random.choice(np.arange(0.1, 1.1, 0.1))) for j in range(self.num_commands)]

        # Reset of vehicles information
        self.vehicles = [Vehicle(f'V{i+1}', np.random.choice(np.arange(0, 1.1, 0.1), size=2), np.random.choice(np.arange(0.1, 1.1, 0.2)), np.random.choice(np.arange(0.1, 1.1, 0.1)), np.random.choice(np.arange(0.2, 1.1, 0.4)), np.ones(3) / 3) for i in range(self.num_vehicles)]

        # Matrix of weights of the commands based in different preference of types of vechicle
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

        # Collect all observations in separate arrays
        vehicle_positions = np.array([vehicle.position for vehicle in self.vehicles]).flatten()
        command_positions = np.array([command.position for command in self.commands]).flatten()
        vehicle_hydrogen = np.array([vehicle.hydrogen for vehicle in self.vehicles])
        vehicle_remaining_working_time = np.array([vehicle.remaining_working_time for vehicle in self.vehicles])
        vehicle_quality_of_service = np.array([vehicle.quality_of_service for vehicle in self.vehicles])
        command_prices = np.array([command.price for command in self.commands])
        command_duration = np.array([command.duration for command in self.commands])
        command_weights = np.array([command.weights for command in self.commands]).flatten()

        # Build the complete observation for each vehicle
        full_observation = np.concatenate([
            vehicle_positions,
            command_positions,
            vehicle_hydrogen,
            vehicle_remaining_working_time,
            vehicle_quality_of_service,
            command_prices,
            command_duration,
            command_weights

        ])

        observations = {}
        for i in range(len(self.vehicles)):
            observations[i] = full_observation

        # Initial observation feedback (concatenates information from vehicles and commands)         
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

        self.match_assignments_vehicule = AssignmentsVehicle(self.commands, self.vehicles)
        assignments_vehicule = self.match_assignments_vehicule.match()

        # Reward calculation for each vehicle
        rewards = []
        for i, vehicule in enumerate(self.vehicles):
            # Check if the vehicle has a preference
            if vehicule.preference:
                # Iterate over the list of preferences to find the position of the current reward
                current_preference_position = None
                for j, (cmd_name, score) in enumerate(vehicule.preference):
                    if cmd_name == vehicule.job.name:
                        current_preference_position = j
                        break

                # Assigns the reward based on the position in the preferences list
                if current_preference_position == 0:
                    rewards.append(10)  # If it's the favorite
                elif current_preference_position == 1:
                    rewards.append(-20)   # If it's the second most preferred
                elif current_preference_position == 2:
                    rewards.append(-50)   # If it's the third most preferred
                else:
                    rewards.append(-100)  # Other cases (less preferred)

        # Vehicles pick up the command position
        for vehicule in self.vehicles:
                vehicule.position = vehicule.job.position

        # Vehicles lose hydrogen after a service
        for vehicule in self.vehicles:
            #if vehicule.job:
                vehicule.hydrogen = vehicule.hydrogen - (vehicule.job.duration * 0.2)
                # Ensure hydrogen does not go below zero
                vehicule.hydrogen = max(vehicule.hydrogen, 0.0)

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
        #self.match_assignments_vehicule = AssignmentsVehicle(self.commands, self.vehicles)
        #assignments_vehicule = self.match_assignments_vehicule.match()
        # print("Matches:", assignments_vehicule)

        # Reset for new commands
        for command in self.commands:
                command.poistion = np.random.choice(np.arange(0, 1.1, 0.1), size=2)
                command.price = np.random.choice(np.arange(0.2, 1.1, 0.1))
                command.duration = np.random.choice(np.arange(0.1, 1.1, 0.1))

        # Matrix of weights of the commands based in different preference of types of vechicle
        commands_weights = []
        
        for _ in range(self.num_commands):
            
            line = [1, 1, 0, 0]

            np.random.shuffle(line)
            
            commands_weights.append(line)

            np.array(commands_weights)
        
        random.shuffle(commands_weights)

        for i, command in enumerate(self.commands):
          command.weights = commands_weights[i]

        done = {i: False for i in range(self.num_vehicles)}

        # Reset preferences (vehicles and commands)
        for vehicle in self.vehicles:
            vehicle.preference = []
        for command in self.commands:
            command.preference = []


        # Collect all observations in separate arrays
        vehicle_positions = np.array([vehicle.position for vehicle in self.vehicles]).flatten()
        command_positions = np.array([command.position for command in self.commands]).flatten()
        vehicle_hydrogen = np.array([vehicle.hydrogen for vehicle in self.vehicles])
        vehicle_remaining_working_time = np.array([vehicle.remaining_working_time for vehicle in self.vehicles])
        vehicle_quality_of_service = np.array([vehicle.quality_of_service for vehicle in self.vehicles])
        command_prices = np.array([command.price for command in self.commands])
        command_duration = np.array([command.duration for command in self.commands])
        command_weights = np.array([command.weights for command in self.commands]).flatten()

        # Build the complete observation for each vehicle
        full_observation = np.concatenate([
            vehicle_positions,
            command_positions,
            vehicle_hydrogen,
            vehicle_remaining_working_time,
            vehicle_quality_of_service,
            command_prices,
            command_duration,
            command_weights

        ])

        observations = {}
        for i in range(len(self.vehicles)):
            observations[i] = full_observation

        # Return the current observation, reward, breakpoint and additional information
        return observations, rewards, done
    
def calculate_distance(position1, position2):
    return math.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2)

def calculate_vehicle_score(command, weights, position):
    score = (command.price * weights[0]) - (calculate_distance(command.position, position) * weights[1]) - (command.duration * weights[2])
    return score

def calculate_command_score(vehicle, weights, position):
    score = ((vehicle.hydrogen * weights[0]) - (calculate_distance(vehicle.position, position) * weights[1]) + (vehicle.remaining_working_time * weights[2]) + (vehicle.quality_of_service * weights[3]))
    return score