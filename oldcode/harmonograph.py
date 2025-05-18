import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os
from datetime import datetime

class Harmonograph:
    def __init__(self, genes=None):
        self.genes = genes if genes else [random.random() for _ in range(20)]
        self.points = []
        self.phenotype = None
        self.time_max = 150
        self.time_step = 0.025
        self.fitness = 0

    def get_copy(self):
        return Harmonograph(self.genes.copy())

    def mutate(self, mutation_rate=0.05):
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                self.genes[i] = min(1, max(0, self.genes[i] + random.uniform(-0.1, 0.1)))
        self.phenotype = None

    def calculate_points(self):
        g = self.genes
        a1 = 300 * (0.15 + 0.1 * g[0])
        a2 = 300 * (0.15 + 0.1 * g[1])
        a3 = 300 * (0.15 + 0.1 * g[2])
        a4 = 300 * (0.15 + 0.1 * g[3])
        v = [-0.02 + 0.04 * g[i] for i in range(4, 8)]
        f = [v[i] + 1 + int(5 * g[8 + i]) for i in range(4)]
        p = [2 * math.pi * g[i] for i in range(12, 16)]
        d = [0.01 * g[i] for i in range(16, 20)]

        self.points.clear()
        t_values = np.arange(0, self.time_max, self.time_step)
        for t in t_values:
            x = a1 * math.sin(t * f[0] + p[0]) * math.exp(-d[0] * t) + a2 * math.sin(t * f[1] + p[1]) * math.exp(-d[1] * t)
            y = a3 * math.sin(t * f[2] + p[2]) * math.exp(-d[2] * t) + a4 * math.sin(t * f[3] + p[3]) * math.exp(-d[3] * t)
            self.points.append([x, y])

    def render(self, save_path=None, resolution=2000):
        self.calculate_points()
        points = np.array(self.points)

        fig, ax = plt.subplots(figsize=(10, 10), dpi=resolution//200)
        ax.plot(points[:, 0], points[:, 1], color='black', linewidth=0.5)
        ax.axis('off')
        ax.set_aspect('equal')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        else:
            plt.show()

    def export(self):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_path = f"outputs/{timestamp}"
        self.render(save_path=output_path + ".png")

        with open(output_path + ".txt", "w") as f:
            f.writelines([f"{g}\n" for g in self.genes])