from harmonograph import Harmonograph

class Population:
    def __init__(self, size):
        self.members = [Harmonograph() for _ in range(size)]

    def evolve(self):
        self.members.sort(key=lambda h: h.fitness, reverse=True)
        top_half = self.members[:len(self.members)//2]
        new_members = []

        for h in top_half:
            child = h.get_copy()
            child.mutate()
            new_members.append(child)

        self.members = top_half + new_members

    def render_all(self):
        for i, h in enumerate(self.members):
            h.export()