import random
import math
from p5 import *

class Harmonograph:
    def __init__(self, genes_init=None):
        self.genes = [random.random() for _ in range(20)] if genes_init is None else genes_init
        self.fitness = 0
        self.time_max = 150
        self.time_step = 0.025
        self.points = []
        self.phenotype = None

    def randomize(self):
        self.genes = [random.random() for _ in range(20)]
        self.phenotype = None

    def uniform_crossover(self, partner):
        child1 = self.get_copy()
        child2 = partner.get_copy()

        for i in range(len(child1.genes)):
            if random.random() < 0.5:
                child1.genes[i], child2.genes[i] = child2.genes[i], child1.genes[i]

        return [child1, child2]

    def mutate(self, mutation_rate=0.05):
        for i in range(len(self.genes)):
            if random.random() <= mutation_rate:
                self.genes[i] = max(0, min(1, self.genes[i] + random.uniform(-0.1, 0.1)))
        self.phenotype = None

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

    def get_copy(self):
        copy = Harmonograph(self.genes[:])
        copy.fitness = self.fitness
        return copy

    def get_phenotype(self, resolution):
        if self.phenotype and self.phenotype.shape[0] == resolution:
            return self.phenotype

        # Create a PGraphics object to render the harmonograph
        canvas = create_graphics(resolution, resolution, renderer='skia')
        canvas.begin_draw()


        self.render(canvas, canvas.width / 2, canvas.height / 2, canvas.width, canvas.height)
        canvas.end_draw()
        self.phenotype = canvas
        return canvas

    def render(self, canvas, x, y, w, h):
        self.calculate_points(w, h)
        canvas.push_matrix()
        canvas.translate(x, y)
        canvas.begin_shape()
        for point in self.points:
            canvas.vertex(point.x, point.y)
        canvas.end_shape()
        canvas.pop_matrix()

    def render_points(self, canvas, x, y, w, h):
        self.calculate_points(w, h)
        canvas.push_matrix()
        canvas.translate(x, y)
        for point in self.points:
            canvas.point(point.x, point.y)
        canvas.pop_matrix()

    def calculate_points(self, w, h):
        a1 = w * (0.15 + 0.1 * self.genes[0])
        a2 = w * (0.15 + 0.1 * self.genes[1])
        a3 = h * (0.15 + 0.1 * self.genes[2])
        a4 = h * (0.15 + 0.1 * self.genes[3])
        v1 = -0.02 + 0.04 * self.genes[4]
        v2 = -0.02 + 0.04 * self.genes[5]
        v3 = -0.02 + 0.04 * self.genes[6]
        v4 = -0.02 + 0.04 * self.genes[7]
        f1 = v1 + 1 + int(5 * self.genes[8])
        f2 = v2 + 1 + int(5 * self.genes[9])
        f3 = v3 + 1 + int(5 * self.genes[10])
        f4 = v4 + 1 + int(5 * self.genes[11])
        p1 = 2 * math.pi * self.genes[12]
        p2 = 2 * math.pi * self.genes[13]
        p3 = 2 * math.pi * self.genes[14]
        p4 = 2 * math.pi * self.genes[15]
        d1 = 0.01 * self.genes[16]
        d2 = 0.01 * self.genes[17]
        d3 = 0.01 * self.genes[18]
        d4 = 0.01 * self.genes[19]

        self.points.clear()

        for t in range(int(self.time_max / self.time_step)):
            t *= self.time_step
            point_x = a1 * math.sin(t * f1 + p1) * math.exp(-d1 * t) + a2 * math.sin(t * f2 + p2) * math.exp(-d2 * t)
            point_y = a3 * math.sin(t * f3 + p3) * math.exp(-d3 * t) + a4 * math.sin(t * f4 + p4) * math.exp(-d4 * t)
            self.points.append(PVector(point_x, point_y))

    def export(self):
        output_filename = f"{year()}-{month():02d}-{day():02d}-{hour():02d}-{minute():02d}-{second():02d}"
        output_path = sketch_path(f"outputs/{output_filename}")
        print(f"Exporting harmonograph to: {output_path}")

        self.get_phenotype(2000).save(output_path + ".png")

        pdf = create_graphics(500, 500, PDF, output_path + ".pdf")
        pdf.begin_draw()
        pdf.no_fill()
        pdf.stroke_weight(pdf.height * 0.001)
        pdf.stroke(0)
        self.render(pdf, pdf.width / 2, pdf.height / 2, pdf.width, pdf.height)
        pdf.dispose()
        pdf.end_draw()

        output_text_lines = [str(gene) for gene in self.genes]
        save_strings(output_path + ".txt", output_text_lines)
