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
MIN_HYDROGEN = 80
MAX_HYDROGEN = 5000
QUANTITY_OF_HYDROGEN_CONSUMPTION_PER_MINUTE = 80

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
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_vehicles, 4), dtype=np.float32)

        # Initialization of vehicles information
        self.vehicles = self._initialize_vehicles()

        # Initialization of commands information
        self.commands = self._initialize_commands()

        # Initialization of the AssignmentsVehicle class
        self.match_assignments_vehicle = AssignmentsVehicle(self.commands, self.vehicles)

    # Initialize the vehicles
    def _initialize_vehicles(self):
        return [Vehicle(
            f'V{i+1}', np.random.choice(np.arange(MIN_GRID_SIZE, MAX_GRID_SIZE, 1), size=2), 
            np.random.choice(np.arange(MIN_HYDROGEN, MAX_HYDROGEN, 50)), 
            np.random.choice(np.arange(MIN_WORKING_TIME, MAX_WORKING_TIME, 10)), 
            np.random.choice(np.arange(MIN_QUALITY_OF_SERVICE, MAX_QUALITY_OF_SERVICE, 1)), 
            int(BAD_TRAFFIC_CONDITION), np.ones(4) / 4) 
            for i in range(self.num_vehicles)]

    # Initialize the commands
    def _initialize_commands(self):
        return [Command(
            f'C{j+1}', np.random.choice(np.arange(MIN_GRID_SIZE, MAX_GRID_SIZE, 1), size=2), 
            np.random.choice(np.arange(MIN_PRICE, MAX_PRICE, 5)), 
            np.random.choice(np.arange(MIN_DURATION, MAX_DURATION, 1)), 
            int(BAD_TRAFFIC_CONDITION),
            np.random.choice([0, 1])
        ) for j in range(self.num_commands)]

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

        command_traffic_condition = np.zeros(self.max_commands)
        command_traffic_condition[:self.num_commands] = np.array([int(command.is_bad_traffic_condition) for command in self.commands])

        command_urgency = np.zeros(self.max_commands)
        command_urgency[:self.num_commands] = np.array([command.urgency for command in self.commands])

        num_commands = np.array([self.num_commands / self.max_commands])

        vehicle_matched = np.array([int(vehicle.is_matched)])

        vehicle_bad_traffic_condition = np.array([int(vehicle.is_bad_traffic_condition)])

        # Calculate distances from this vehicle to each command
        distances = np.zeros(self.max_commands)
        for j, c in enumerate(self.commands):
            distances[j] = calculate_distance(vehicle.position, c.position) / MAX_DISTANCE

        # Get command weights and fill with zero value for non-existing commands
        command_weights = np.full((self.max_commands, 4), 0.)
        for j, c in enumerate(self.commands):
            command_weights[j] = c.weights

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
            command_traffic_condition,
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
        self.commands = self._initialize_commands()

        # Update the weights of the commands
        for c in self.commands:
            c.weights = np.random.rand(4)

        # Update the traffic condition for each command
        for c in self.commands:
            c.is_bad_traffic_condition = self.is_bad_traffic_area(c.position)

        # Reset of vehicles information
        self.vehicles = self._initialize_vehicles()

        # Update the traffic condition for each vehicle
        for v in self.vehicles:
            v.is_bad_traffic_condition = self.is_bad_traffic_area(v.position)

        # Reset the assignments
        self.match_assignments_vehicle.reset()

        # Reset preferences (vehicles and commands)
        for v in self.vehicles:
            v.preference = []
        for c in self.commands:
            c.preference = []

        observations = {}
        for i in range(len(self.vehicles)):
            observations[i] = self._get_observation(i)

        return observations

    def step(self, actions):
        # Update vehicle weights according to action taken
        for i, v in enumerate(self.vehicles):
            v.weights = actions[i]

        # Updates the preference of each vehicle and order with name and Score
        for v in self.vehicles:
            for c in self.commands:
                score = calculate_vehicle_score(c, v.weights, v.position)
                v.preference.append((c.name, score))

        # Updates the preference of each command and order with name and Score
        for c in self.commands:
            for v in self.vehicles:
                score = calculate_command_score(v, c.weights, c.position)
                c.preference.append((v.name, score))

        # Rank the preference of each vehicle and order according to score
        for v in self.vehicles:
            v.preference.sort(key=lambda x: x[1], reverse=True)

        # Rank the preference of each command and order according to score
        for c in self.commands:
            c.preference.sort(key=lambda x: x[1], reverse=True)

        # Get the list of available vehicles
        vehicles_for_assigment = []
        for v in self.vehicles:
            if not v.is_matched and v.hydrogen > 0 and v.remaining_working_time > 0:
                vehicles_for_assigment.append(v)

        # Match vehicles with commands
        self.match_assignments_vehicle = AssignmentsVehicle(list(self.commands), vehicles_for_assigment)
        assignments_vehicle = self.match_assignments_vehicle.match()

        # Reward calculation for each vehicle
        rewards = []
        for i, v in enumerate(self.vehicles):
            # Check if the vehicle has a job assigned
            if v.job is not None:
                # Find the score for the matched command
                for cmd_name, score in v.preference:
                    if cmd_name == v.job.name:
                        #print(f"Vehicle {v.name} matched with command {v.job.name} with score {score}")
                        rewards.append(score)
                        break
            else:
                rewards.append(0)  # If there is no job assigned

        # Vehicles pick up the command position, update the traffic condition, quantity of hydrogen, and remaining working time
        for v in self.vehicles:
            if v.job is not None:
                v.position = v.job.position
                v.is_bad_traffic_condition = self.is_bad_traffic_area(v.position)
                v.remaining_working_time -= v.job.duration
                if v.remaining_working_time < 0:
                    v.remaining_working_time = 0

        # Vehicles lose hydrogen after a service (does not go less than level 0)
        for v in self.vehicles:
            if v.job is not None:
                v.hydrogen -= v.job.duration * QUANTITY_OF_HYDROGEN_CONSUMPTION_PER_MINUTE  # 80 is the hydrogen consumption rate (80 units per minute)
                if v.hydrogen < 0:
                    v.hydrogen = 0

        # Vehicles are not available for a new job if the duration is greater than 10 minutes in the next step
        for v in self.vehicles:
            if v.job is not None and v.job.duration > TIME_LIMIT_FOR_THE_VEHICLE_NOT_TO_BE_DEACTIVATED:
                v.is_matched = True
            else:
                v.is_matched = False

        # Set the number of commands for the next step
        self.num_commands = np.random.poisson(lam=self.num_vehicles)

        # Create commands with different information for the next step
        self.commands = self._initialize_commands()

        # Update the traffic condition for each command
        for c in self.commands:
            c.is_bad_traffic_condition = self.is_bad_traffic_area(c.position)

        # Update the weights of the commands
        for c in self.commands:
            c.weights = np.random.rand(4)

        done = {i: True if v.hydrogen <= 0 or v.remaining_working_time <= 0 else False for i, v in enumerate(self.vehicles)}  # If a vehicle runs out of hydrogen or its remaining working time is over, the episode ends for it

        # Reset preferences (vehicles and commands)
        for v in self.vehicles:
            v.preference = []
        for c in self.commands:
            c.preference = []

        # Reset the assignments
        self.match_assignments_vehicle.reset()

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
    traffic_condition = command.is_bad_traffic_condition
    normalized_price = command.price / MAX_PRICE
    normalized_distance = calculate_distance(command.position, position) / MAX_DISTANCE
    normalized_duration = command.duration / MAX_DURATION

    # If the weight is zero, it is replaced by 1, otherwise it is multiplied by 2
    weights = np.where(weights == 0, 1, weights + 1)

    # If the traffic condition is bad, the distance is multiplied by 1.5 (mean velocity decrease by 50%)
    if traffic_condition:
        score = ((normalized_price * weights[0]) - (DISTANCE_RATE_IF_BAD_TRAFFIC * normalized_distance * weights[1]) - (normalized_duration * weights[2]) + (command.urgency * weights[3]))
    else:
        score = ((normalized_price * weights[0]) - (normalized_distance * weights[1]) - (normalized_duration * weights[2]) + (command.urgency * weights[3]))
    
    return score


def calculate_command_score(vehicle, weights, position):
    # Normalize each attribute
    normalized_hydrogen = vehicle.hydrogen / MAX_HYDROGEN
    traffic_condition = vehicle.is_bad_traffic_condition
    normalized_distance = calculate_distance(vehicle.position, position) / MAX_DISTANCE
    normalized_working_time = vehicle.remaining_working_time / MAX_WORKING_TIME
    normalized_quality_of_service = vehicle.quality_of_service / MAX_QUALITY_OF_SERVICE

    # If the traffic condition is bad, the distance is multiplied by 1.5 (mean velocity decrease by 50%)
    if traffic_condition == True:
        score = ((normalized_hydrogen * weights[0]) - (DISTANCE_RATE_IF_BAD_TRAFFIC * normalized_distance * weights[1]) + (normalized_working_time * weights[2]) + (normalized_quality_of_service * weights[3]))
    else:
        score = ((normalized_hydrogen * weights[0]) - (normalized_distance * weights[1]) + (normalized_working_time * weights[2]) + (normalized_quality_of_service * weights[3]))
    
    return score