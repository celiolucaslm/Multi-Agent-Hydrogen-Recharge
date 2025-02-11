from vehicle import Vehicle
from command import Command
import math

# Declaration Class Assignments (Gale-Shapley)
class AssignmentsVehicle:
    def __init__(self, commands, vehicles):
        self.assignments = {}
        self.commands = commands
        self.vehicles = vehicles
        self.assignment_count = 0

        for command in commands:
            self.assignments[command.name] = command

        for vehicle in vehicles:
            self.assignments[vehicle.name] = vehicle

    def assign(self, vehicle_name, command_name):
        command = self.assignments[command_name]
        vehicle = self.assignments[vehicle_name]

        command.is_matched = True
        command.vehicule = vehicle

        vehicle.is_matched = True
        vehicle.job = command

        self.assignment_count += 1

    def unassign(self, vehicle_name, command_name):
        command = self.assignments[command_name]
        vehicle = self.assignments[vehicle_name]

        command.is_matched = False
        command.vehicule = None

        vehicle.is_matched = False
        vehicle.job = None

        self.assignment_count -= 1

    def reset(self):
        for command in self.commands:
            command.is_matched = False
            command.vehicule = None
            command.reward = 0
            command.reset_index()

        for vehicle in self.vehicles:
            vehicle.is_matched = False
            vehicle.job = None
            vehicle.reward = 0
            vehicle.reset_index()

    def match(self):
        proposals = {command.name: [] for command in self.commands}

        # Loop until all vehicles are matched
        while True:
            # Find all unmatched vehicles
            unmatched_vehicles = [vehicle for vehicle in self.vehicles if not vehicle.is_matched]
            unmatched_commands = [command for command in self.commands if not command.is_matched]
            if not unmatched_vehicles:
                break

            if unmatched_commands:
                for vehicle in unmatched_vehicles:
                        # Vehicule makes a proposal to the next commande in its preference list
                        command_name, score = vehicle.propose()
                        command = self.assignments[command_name]
                        command.update_score(score)

                        proposals[command_name].append((vehicle, score))
            else:
                break
                     

            # Process proposals for each commande
            for command_name, proposers in proposals.items():
                if proposers:
                    proposers.sort(key=lambda x: x[1], reverse=True)  # Sort proposers by score
                    best_proposer, best_score = proposers[0]
                    command = self.assignments[command_name]

                    if command.is_available():
                        self.assign(best_proposer.name, command_name)
                    else:
                        current_vehicule = command.vehicule
                        current_vehicule_score = next((score for v, score in proposers if v.name == current_vehicule.name), None)

                        # Check if current_vehicule_score is different from None before making the comparison
                        if current_vehicule_score is not None and best_score > current_vehicule_score:
                            self.unassign(current_vehicule.name, command_name)
                            self.assign(best_proposer.name, command_name)


            # Clear proposals after processing
            proposals = {command.name: [] for command in self.commands}

        return self.sets()

    def sets(self):
        matches = {}
        for i in self.assignments:
            assignment = self.assignments[i]
            if isinstance(assignment, Vehicle) and assignment.is_matched:
                matches[frozenset([assignment.name, assignment.job.name])] = True
        return list(matches.keys())
    
# --- Test the AssignmentsVehicle class ---

# Auxiliary functions to calculate the score and distance of a vehicle and a command
def calculate_distance(position1, position2):
    return math.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2)

def calculate_vehicle_score(command, weights, position):
    score = (command.price * weights[0]) - (calculate_distance(command.position, position) * weights[1]) - (command.duration * weights[2])
    return score

def calculate_command_score(vehicle, weights, position):
    score = ((vehicle.hydrogen * weights[0]) - (calculate_distance(vehicle.position, position) * weights[1]) + (vehicle.remaining_working_time * weights[2]) + (vehicle.quality_of_service * weights[3]))
    return score

# ---------------------------------------------------------------------

# # Create a list of commands
# for i in range(1, 4):
#     command1 = Command(name='command1', position=(0, 0), price=10, duration=2)
#     command2 = Command(name='command2', position=(10, 10), price=20, duration=3)

#     commands = [command1, command2]

#     # Create a list of vehicles
#     vehicle1 = Vehicle(name='vehicle1', position=(0, 0), hydrogen=100, remaining_working_time=10, quality_of_service=1, weights=[1, 1, 1])
#     vehicle2 = Vehicle(name='vehicle2', position=(10, 10), hydrogen=100, remaining_working_time=10, quality_of_service=1, weights=[1, 1, 1])
#     vehicle3 = Vehicle(name='vehicle3', position=(20, 20), hydrogen=100, remaining_working_time=10, quality_of_service=1, weights=[1, 1, 1])
#     vehicles = [vehicle1, vehicle2, vehicle3]

#     # Updates the preference of each vehicle and order with name and Score
#     for vehicule in vehicles:
#         for commande in commands:
#             score = calculate_vehicle_score(commande, vehicule.weights, vehicule.position)
#             vehicule.preference.append((commande.name, score))

#     for commande in commands:
#         for vehicule in vehicles:
#             score = calculate_command_score(vehicule, commande.weights, commande.position)
#             commande.preference.append((vehicule.name, score))

#     # Rank the preference of each vehicle and order according to score
#     for vehicule in vehicles:
#         vehicule.preference.sort(key=lambda x: x[1], reverse=True)

#     for commande in commands:
#         commande.preference.sort(key=lambda x: x[1], reverse=True)

#     # Create an instance of the AssignmentsVehicle class
#     GaleShapley = AssignmentsVehicle(commands, vehicles)

#     # Match the vehicles with the commands
#     matches = GaleShapley.match()
#     print(matches)