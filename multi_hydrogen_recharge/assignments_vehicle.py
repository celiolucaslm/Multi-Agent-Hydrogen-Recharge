from vehicle import Vehicle

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
            if not unmatched_vehicles:
                break

            for vehicle in unmatched_vehicles:
                # Vehicule makes a proposal to the next commande in its preference list
                command_name, score = vehicle.propose()
                command = self.assignments[command_name]
                command.update_score(score)

                proposals[command_name].append((vehicle, score))

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