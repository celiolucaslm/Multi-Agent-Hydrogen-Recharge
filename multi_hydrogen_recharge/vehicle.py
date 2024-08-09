# Declaration Class Vehicle
class Vehicle:
    def __init__(self, name, position, hydrogen, remaining_working_time, quality_of_service, weights):
        self.name = name
        self.position = position
        self.hydrogen = hydrogen
        self.remaining_working_time = remaining_working_time
        self.quality_of_service = quality_of_service
        self.weights = weights
        self.preference = []
        self.is_matched = False
        self.job = None
        self.index = 0
        self.score = 0

    def propose(self):
        commande = self.preference[self.index]
        self.index += 1
        return commande

    def is_available(self):
        return not self.is_matched

    def update_score(self, new_score):
        self.score = new_score

    def reset_index(self):
        self.index = 0