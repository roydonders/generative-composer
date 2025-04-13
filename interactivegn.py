from p5 import *
import math
from population import Population

pop = None
cells = []
hovered_indiv = None
population_size = 100  # Define your population size
resolution = 50        # Assuming resolution is a global variable used for phenotype rendering


def settings():
    size(int(display_width * 0.9), int(display_height * 0.8), P2D)
    smooth(8)


def setup():
    global pop, cells
    pop = Population(population_size=population_size)
    cells = calculate_grid(population_size, 0, 0, width, height - 30, 30, 10, 30, True)
    #text_size(constrain(cells[0][0].z * 0.15, 11, 14))


def draw():
    global hovered_indiv
    background(235)
    hovered_indiv = None
    row, col = 0, 0

    for i in range(pop.get_size()):
        x = cells[row][col].x
        y = cells[row][col].y
        d = cells[row][col].z

        no_stroke()
        fill(0)

        if x < mouse_x < x + d and y < mouse_y < y + d:
            hovered_indiv = pop.get_indiv(i)
            rect((x - 1, y - 1), d + 2, d + 2)

        if pop.get_indiv(i).get_fitness() > 0:
            rect((x - 3, y - 3), d + 6, d + 6)

        # Draw phenotype
        image(pop.get_indiv(i).get_phenotype(resolution), (x, y), (d, d))

        # Draw fitness text
        fill(0)
        text_align("CENTER", "TOP")
        text(f"{pop.get_indiv(i).get_fitness():.2f}", (x + d / 2, y + d + 5))

        col += 1
        if col >= len(cells[row]):
            row += 1
            col = 0

    # Control instructions
    fill(128)
    text_size(14)
    text_align("LEFT", "BOTTOM")
    text("Controls:     [click over indiv] set as preferred     [enter] evolve     [r] reset     [e] export individ hovered by the cursor", (30, height - 30))


def key_released(event):
    global hovered_indiv
    if event.key == 'ENTER':
        pop.evolve()
    elif event.key == ' ':
        pop.evolve()
    elif event.key == 'r':
        pop.initialize()
    elif event.key == 'e':
        if hovered_indiv is not None:
            hovered_indiv.export()
    else:
        if hovered_indiv is not None:
            fit = hovered_indiv.get_fitness()
            if event.key == 'UP':
                fit = min(fit + 0.1, 1)
            elif event.key == 'DOWN':
                fit = max(fit - 0.1, 0)
            elif event.key == 'RIGHT':
                fit = 1
            elif event.key == 'LEFT':
                fit = 0
            hovered_indiv.set_fitness(fit)


def mouse_released():
    global hovered_indiv
    if hovered_indiv is not None:
        hovered_indiv.set_fitness(1 if hovered_indiv.get_fitness() < 1 else 0)


def calculate_grid(cells, x, y, w, h, margin_min, gutter_h, gutter_v, align_top):
    cols, rows, cell_size = 0, 0, 0
    while cols * rows < cells:
        cols += 1
        cell_size = ((w - margin_min * 2) - (cols - 1) * gutter_h) / cols
        rows = math.floor((h - margin_min * 2) / (cell_size + gutter_v))
    if cols * (rows - 1) >= cells:
        rows -= 1
    margin_hor_adjusted = ((w - cols * cell_size) - (cols - 1) * gutter_h) / 2
    if rows == 1 and cols > cells:
        margin_hor_adjusted = ((w - cells * cell_size) - (cells - 1) * gutter_h) / 2
    margin_ver_adjusted = ((h - rows * cell_size) - (rows - 1) * gutter_v) / 2
    if align_top:
        margin_ver_adjusted = min(margin_hor_adjusted, margin_ver_adjusted)

    positions = []
    for row in range(rows):
        row_y = y + margin_ver_adjusted + row * (cell_size + gutter_v)
        row_positions = []
        for col in range(cols):
            col_x = x + margin_hor_adjusted + col * (cell_size + gutter_h)
            row_positions.append(Vector(col_x, row_y, cell_size))
        positions.append(row_positions)

    return positions


if __name__ == '__main__':
    run(renderer='skia')
