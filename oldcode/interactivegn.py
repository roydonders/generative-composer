import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.patches as patches
from population import Population

# Ensure interactive matplotlib
matplotlib.use('TkAgg')


# Grid layout calculation

def calculate_grid(n_cells, width, height, margin=30, gutter_h=10, gutter_v=30):
    cols = 0
    rows = 0
    cell_size = 0
    while cols * rows < n_cells:
        cols += 1
        cell_size = ((width - margin * 2) - (cols - 1) * gutter_h) / cols
        rows = int((height - margin * 2) / (cell_size + gutter_v))

    if cols * (rows - 1) >= n_cells:
        rows -= 1

    margin_hor_adjusted = ((width - cols * cell_size) - (cols - 1) * gutter_h) / 2
    margin_ver_adjusted = ((height - rows * cell_size) - (rows - 1) * gutter_v) / 2

    positions = []
    for row in range(rows):
        row_y = margin_ver_adjusted + row * (cell_size + gutter_v)
        for col in range(cols):
            col_x = margin_hor_adjusted + col * (cell_size + gutter_h)
            positions.append((col_x, row_y, cell_size))
    return positions

# Main visualization

def draw_population(population):
    width, height = 1200, 800
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    plt.subplots_adjust(bottom=0.1)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')

    cells = calculate_grid(len(population.members), width, height - 50)
    hovered_indiv = [None]  # Use list to allow mutation in closures

    def draw_all():
        ax.clear()
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis('off')
        for i, (x, y, d) in enumerate(cells):
            if i >= len(population.members):
                break
            indiv = population.members[i]
            indiv.calculate_points()
            pts = np.array(indiv.points)
            ax.plot(pts[:, 0] * (d / 600) + x + d / 2, pts[:, 1] * (d / 600) + y + d / 2, lw=0.5, color='black')
            ax.text(x + d / 2, y + d + 10, f"{indiv.fitness:.2f}", ha='center', fontsize=8)
        fig.canvas.draw_idle()

    def on_click(event):
        for i, (x, y, d) in enumerate(cells):
            if x < event.x < x + d and y < event.y < y + d:
                hovered_indiv[0] = population.members[i]
                f = hovered_indiv[0].fitness
                hovered_indiv[0].fitness = 0 if f >= 1 else 1
                draw_all()
                break

    def on_key(event):
        nonlocal population
        print(f"Key pressed: {event.key}")
        key = event.key.lower()

        if key in ['enter', ' ']:
            population.evolve()
            draw_all()
        elif key == 'r':
            newpopulation = Population(16)
            #todo fix hardcoding

            draw_all(newpopulation)
        elif key == 'e' and hovered_indiv[0]:
            hovered_indiv[0].export()
        elif hovered_indiv[0]:
            fit = hovered_indiv[0].fitness
            if key == 'up':
                hovered_indiv[0].fitness = min(fit + 0.1, 1)
            elif key == 'down':
                hovered_indiv[0].fitness = max(fit - 0.1, 0)
            elif key == 'right':
                hovered_indiv[0].fitness = 1
            elif key == 'left':
                hovered_indiv[0].fitness = 0
            draw_all()

    fig.canvas.mpl_connect('button_release_event', on_click)
    fig.canvas.mpl_connect('key_release_event', on_key)
    draw_all()
    ax.text(30, height - 10, "Controls: [click over indiv] set as preferred     [enter] evolve     [r] reset     [e] export hovered", fontsize=10)
    plt.show()


if __name__ == '__main__':
    pop = Population(16)
    draw_population(pop)
