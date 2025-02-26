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
            #vehicle.is_matched = False
            vehicle.job = None
            vehicle.reward = 0
            vehicle.reset_index()

    def match(self):
        proposals = {vehicle.name: [] for vehicle in self.vehicles}

        # Loop until all vehicles are matched if we have enough commands
        while True:
            # Find all unmatched vehicles and commands
            unmatched_vehicles = [vehicle for vehicle in self.vehicles if not vehicle.is_matched]
            unmatched_commands = [command for command in self.commands if not command.is_matched]
            if not unmatched_commands:
                break

            if unmatched_vehicles:
                for command in unmatched_commands:
                        # Command makes a proposal to the next vehicle in its preference list
                        vehicle_name, score = command.propose()
                        if vehicle_name in self.assignments:
                            vehicle = self.assignments[vehicle_name]
                            vehicle.update_score(score)

                            proposals[vehicle_name].append((command, score))
            else:
                break
                    

            # Process proposals for each vehicle
            for vehicle_name, proposers in proposals.items():
                if proposers:
                    proposers.sort(key=lambda x: x[1], reverse=True)  # Sort proposers by score
                    best_proposer, best_score = proposers[0]
                    vehicle = self.assignments[vehicle_name]

                    if vehicle.is_available():
                        self.assign(vehicle_name, best_proposer.name)
                    else:
                        current_command = vehicle.command
                        if current_command is not None:
                            current_command_score = next((score for c, score in proposers if c.name == current_command.name), None)
                    
                            # Check if current_command_score is different from None before making the comparison
                            if current_command_score is not None and best_score > current_command_score:
                                self.unassign(vehicle_name, current_command.name)
                                self.assign(vehicle_name, best_proposer.name)


            # Clear proposals after processing
            proposals = {vehicle.name: [] for vehicle in self.vehicles}

        return self.sets()

    def sets(self):
        matches = {}
        for i in self.assignments:
            assignment = self.assignments[i]
            if isinstance(assignment, Vehicle) and assignment.is_matched:
                matches[frozenset([assignment.name, assignment.job.name])] = True
        return list(matches.keys())