from vehicle import Vehicle
from command import Command
from assignments_vehicle import AssignmentsVehicle
import math
import random
from gym import spaces
import numpy as np
from pettingzoo import ParallelEnv

# -----------------------------------------------------------------------
# Constants variables of the environment
# -----------------------------------------------------------------------	
MIN_HYDROGEN = 50
MAX_HYDROGEN = 5000

MIN_WORKING_TIME = 5
MAX_WORKING_TIME = 480

MIN_QUALITY_OF_SERVICE = 1
MAX_QUALITY_OF_SERVICE = 5

MIN_PRICE = 20
MAX_PRICE = 500

MIN_DURATION = 5
MAX_DURATION = 20

BAD_TRAFFIC_CONDITION = False
START_OF_AREA_WITH_BAD_TRAFFIC = 40
END_OF_AREA_WITH_BAD_TRAFFIC = 60
DISTANCE_RATE_IF_BAD_TRAFFIC = 1.5

TIME_LIMIT_FOR_THE_VEHICLE_NOT_TO_BE_DEACTIVATED = 10

MIN_GRID_SIZE = 0
MAX_GRID_SIZE = 100
MAX_DISTANCE = math.sqrt(MIN_GRID_SIZE**2 + MAX_GRID_SIZE**2)  # Positions are within a 100x100 grid

# -----------------------------------------------------------------------
# Declaration MultiHydrogenRecharge (Environment)
# -----------------------------------------------------------------------
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
        self.max_commands = num_vehicles * 4  # Define a maximum number of commands to receive
        self.num_commands = num_commands if num_commands is not None else np.random.poisson(lam=num_vehicles)

        # Defining the action space
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_vehicles, 3), dtype=np.float32)

        # Initialization of vehicles information
        self.vehicles = [Vehicle(f'V{i+1}', np.random.choice(np.arange(MIN_GRID_SIZE, MAX_GRID_SIZE, 1), size=2), np.random.choice(np.arange(MIN_HYDROGEN, MAX_HYDROGEN, 50)), np.random.choice(np.arange(MIN_WORKING_TIME, MAX_WORKING_TIME, 10)), np.random.choice(np.arange(MIN_QUALITY_OF_SERVICE, MAX_QUALITY_OF_SERVICE, 1)), int(BAD_TRAFFIC_CONDITION), np.ones(3) / 3) for i in range(num_vehicles)]

        # Initialization of commands information
        self.commands = [Command(f'C{j+1}', np.random.choice(np.arange(MIN_GRID_SIZE, MAX_GRID_SIZE, 1), size=2), np.random.choice(np.arange(MIN_PRICE, MAX_PRICE, 5)), np.random.choice(np.arange(MIN_DURATION, MAX_DURATION, 1)), int(BAD_TRAFFIC_CONDITION)) for j in range(self.num_commands)]

        # Initialization of the AssignmentsVehicle class
        self.match_assignments_vehicule = AssignmentsVehicle(self.commands, self.vehicles)

    # Define the condition of traffic for a vehilce or command based on the position
    def is_bad_traffic_area(self, position):
        x, y = position
        if START_OF_AREA_WITH_BAD_TRAFFIC <= x <= END_OF_AREA_WITH_BAD_TRAFFIC and START_OF_AREA_WITH_BAD_TRAFFIC <= y <= END_OF_AREA_WITH_BAD_TRAFFIC:
            return True
        else:
            return False 

    # Define the observation space of a vehicle
    def _get_observation(self, vehicle_index):
        # Select the vehicle to send his observation state
        vehicle = self.vehicles[vehicle_index]

        vehicle_position = vehicle.position / MAX_GRID_SIZE # Normalize position
        command_positions = np.zeros((self.max_commands, 2))
        if self.num_commands > 0:
            command_positions[:self.num_commands] = np.array([command.position for command in self.commands]) / MAX_GRID_SIZE  # Normalize position

        vehicle_hydrogen = np.array([vehicle.hydrogen / MAX_HYDROGEN])  

        vehicle_remaining_working_time = np.array([vehicle.remaining_working_time / MAX_WORKING_TIME]) 

        vehicle_quality_of_service = np.array([vehicle.quality_of_service / MAX_QUALITY_OF_SERVICE]) 

        command_prices = np.zeros(self.max_commands)
        command_prices[:self.num_commands] = np.array([command.price for command in self.commands]) / MAX_PRICE

        command_duration = np.zeros(self.max_commands)
        command_duration[:self.num_commands] = np.array([command.duration for command in self.commands]) / MAX_DURATION

        num_commands = np.array([self.num_commands / self.max_commands])

        vehicle_matched = np.array([int(vehicle.is_matched)])

        vehicle_bad_traffic_condition = np.array([int(vehicle.is_bad_traffic_condition)])

        # Calculate distances from this vehicle to each command
        distances = np.zeros(self.max_commands)
        for j, command in enumerate(self.commands):
            distances[j] = calculate_distance(vehicle.position, command.position) / MAX_DISTANCE

        # Get command weights and fill with zero value for non-existing commands
        command_weights = np.full((self.max_commands, 4), 0.)
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
            vehicle_matched,
            vehicle_bad_traffic_condition,
            command_weights.flatten()
        ])

        return observation

    # Reset the environment
    def reset(self):

        # Set the number of commands
        self.num_commands = np.random.poisson(lam=self.num_vehicles)

        # Reset of commands information
        self.commands = [Command(f'C{j+1}', np.random.choice(np.arange(MIN_GRID_SIZE, MAX_GRID_SIZE, 1), size=2), np.random.choice(np.arange(MIN_PRICE, MAX_PRICE, 5)), np.random.choice(np.arange(MIN_DURATION, MAX_DURATION, 1)), int(BAD_TRAFFIC_CONDITION)) for j in range(self.num_commands)]

        # Update the weights of the commands
        for command in self.commands:
            command.weights = np.random.rand(4)

        # Reset of vehicles information
        self.vehicles = [Vehicle(f'V{i+1}', np.random.choice(np.arange(MIN_GRID_SIZE, MAX_GRID_SIZE, 1), size=2), np.random.choice(np.arange(MIN_HYDROGEN, MAX_HYDROGEN, 50)), np.random.choice(np.arange(MIN_WORKING_TIME, MAX_WORKING_TIME, 10)), np.random.choice(np.arange(MIN_QUALITY_OF_SERVICE, MAX_QUALITY_OF_SERVICE, 1)), int(BAD_TRAFFIC_CONDITION), np.ones(3) / 3) for i in range(self.num_vehicles)]

        # Reset the assignments
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

        # Updates the preference of each command and order with name and Score
        for commande in self.commands:
            for vehicule in self.vehicles:
                vehicle.is_bad_traffic_condition = self.is_bad_traffic_area(vehicle.position)
                score = calculate_command_score(vehicule, commande.weights, commande.position)
                commande.preference.append((vehicule.name, score))

        # Rank the preference of each vehicle and order according to score
        for vehicule in self.vehicles:
            vehicule.preference.sort(key=lambda x: x[1], reverse=True)

        # Rank the preference of each command and order according to score
        for commande in self.commands:
            commande.preference.sort(key=lambda x: x[1], reverse=True)

        # Get the list of available vehicles
        vehicles_for_assigment = []
        for vehicle in self.vehicles:
            if not vehicle.is_matched:
                vehicles_for_assigment.append(vehicle)

        # Match vehicles with commands
        self.match_assignments_vehicule = AssignmentsVehicle(list(self.commands), vehicles_for_assigment)
        assignments_vehicule = self.match_assignments_vehicule.match()

        # for c in self.commands:
        #     print(f'Preference of the Command {c.name}:', c.preference)
        
        # for v in self.vehicles:
        #     print(f'Preference of the Vehicle {v.name}:', v.preference)   

        #print('Assignments:', assignments_vehicule)

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
                    rewards.append(-10)  # Other cases (less preferred)
            else:
                rewards.append(0)  # If there is no preference

        # Vehicles pick up the command position
        for vehicule in self.vehicles:
            if vehicule.job is not None:
                vehicule.position = vehicule.job.position

        # Update the traffic condition for each vehicle
        for vehicle in self.vehicles:
            vehicle.is_bad_traffic_condition = self.is_bad_traffic_area(vehicle.position)

        # Vehicles lose hydrogen after a service
        for vehicule in self.vehicles:
            if vehicule.job is not None:
                vehicule.hydrogen = vehicule.hydrogen - (vehicule.job.duration * 80)  # 80 is the hydrogen consumption rate (80 units per minute)

        # Vehicles are not avaliable for a new job if the duration is greater than 10 minutes in the next step
        for vehicle in self.vehicles:
            if vehicle.job is not None and vehicle.job.duration > TIME_LIMIT_FOR_THE_VEHICLE_NOT_TO_BE_DEACTIVATED:
                vehicle.is_matched = True
            else:
                vehicle.is_matched = False

        # Set the number of commands for the next step
        self.num_commands = np.random.poisson(lam=self.num_vehicles)

        # Create commands with different information for the next step
        self.commands = [Command(f'C{j+1}', np.random.choice(np.arange(MIN_GRID_SIZE, MAX_GRID_SIZE, 1), size=2), np.random.choice(np.arange(MIN_PRICE, MAX_PRICE, 5)), np.random.choice(np.arange(MIN_DURATION, MAX_DURATION, 1)), int(BAD_TRAFFIC_CONDITION)) for j in range(self.num_commands)]

        # Update the weights of the commands
        for command in self.commands:
            command.weights = np.random.rand(4)

        done = {i: False for i in range(self.num_vehicles)} # There is no terminal state in the environment

        # Reset preferences (vehicles and commands)
        for vehicle in self.vehicles:
            vehicle.preference = []
        for command in self.commands:
            command.preference = []

        # Reset the assignments
        self.match_assignments_vehicule.reset()

        observations = {}
        for i in range(len(self.vehicles)):
            observations[i] = self._get_observation(i)

        # Return the current observation, reward, and done status
        return observations, rewards, done

# -----------------------------------------------------------------------
# Auxiliary functions
# -----------------------------------------------------------------------
def calculate_distance(position1, position2):
    return math.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2)

def calculate_vehicle_score(command, weights, position):
    # Normalize each attribute
    normalized_price = command.price / MAX_PRICE
    normalized_distance = calculate_distance(command.position, position) / MAX_DISTANCE
    normalized_duration = command.duration / MAX_DURATION

    score = ((normalized_price * weights[0])) - (normalized_distance * weights[1]) - (normalized_duration * weights[2])
    return score

def calculate_command_score(vehicle, weights, position):
    # Normalize each attribute
    normalized_hydrogen = vehicle.hydrogen / MAX_HYDROGEN
    traffic_condition = vehicle.is_bad_traffic_condition
    if traffic_condition == True:
        normalized_distance =  (DISTANCE_RATE_IF_BAD_TRAFFIC * calculate_distance(vehicle.position, position)) / MAX_DISTANCE # If the traffic condition is bad, the distance is multiplied by 1.5 (mean velocity decrease by 50%)
    else:
        normalized_distance = calculate_distance(vehicle.position, position) / MAX_DISTANCE
    normalized_working_time = vehicle.remaining_working_time / MAX_WORKING_TIME
    normalized_quality_of_service = vehicle.quality_of_service / MAX_QUALITY_OF_SERVICE
    
    score = ((normalized_hydrogen * weights[0]) - (normalized_distance * weights[1]) + (normalized_working_time * weights[2]) + (normalized_quality_of_service * weights[3]))
    return score