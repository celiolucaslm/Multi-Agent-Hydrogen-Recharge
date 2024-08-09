# Declaration Class Command
class Command:
    def __init__(self, name, position, price, duration):
        self.name = name
        self.position = position
        self.price = price
        self.duration = duration
        self.weights = [0.25, 0.25, 0.25, 0.25]
        self.preference = []
        self.is_matched = False
        self.vehicle = None
        self.index = 0
        self.score = 0

    def propose(self):
        vehicule = self.preference[self.index]
        self.index += 1
        return vehicule
    
    def is_available(self):
        return not self.is_matched

    def update_score(self, new_score):
        self.score = new_score

    def reset_index(self):
        self.index = 0