import pygame
import random
import os
from midiutil import MIDIFile
import numpy as np

# Basic setup for pygame and GUI
pygame.init()

# Define some constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# MIDI sequence data (for example, pitch numbers)
MIDI_NOTE_RANGE = (48, 84)  # Note range (MIDI notes from 48 to 84)
MIDI_DURATION_CHOICES = [0.5, 0.75, 1]  # Possible durations for each note

# Genetic algorithm parameters
POP_SIZE = 50  # Population size
MUTATION_RATE = 0.1  # Mutation rate
CROSSOVER_RATE = 0.7  # Crossover rate
NUM_GENERATIONS = 100  # Number of generations to evolve

# Create the window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Interactive Genetic Algorithm for MIDI Evolution")

# Fonts for the GUI
font = pygame.font.Font(None, 36)

# Basic MIDI sequence generator for testing
def generate_random_midi_sequence(length=16):
    return [random.randint(MIDI_NOTE_RANGE[0], MIDI_NOTE_RANGE[1]) for _ in range(length)]

# Fitness function: compares a candidate sequence to the reference
def fitness(candidate, reference):
    score = 0
    for c_note, r_note in zip(candidate, reference):
        score -= abs(c_note - r_note)  # Inverse distance score (closer notes are better)
    return score

# Mutation: randomly change a note in the sequence
def mutate(sequence):
    mutated_sequence = sequence[:]
    if random.random() < MUTATION_RATE:
        idx = random.randint(0, len(sequence) - 1)
        mutated_sequence[idx] = random.randint(MIDI_NOTE_RANGE[0], MIDI_NOTE_RANGE[1])
    return mutated_sequence

# Crossover: combine two sequences to create a child sequence
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    return parent1  # No crossover, return parent1

# Generate the next generation using selection, crossover, and mutation
def evolve_population(population, reference):
    scores = [(ind, fitness(ind, reference)) for ind in population]
    scores.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness

    # Create new population through crossover and mutation
    new_population = []
    for i in range(POP_SIZE // 2):
        parent1 = scores[i][0]
        parent2 = scores[i + 1][0]

        # Crossover and mutation
        child1 = crossover(parent1, parent2)
        child2 = crossover(parent2, parent1)

        new_population.append(mutate(child1))
        new_population.append(mutate(child2))

    return new_population

# Function to create and save a MIDI file
def save_midi(sequence, filename="output"):
    midi = MIDIFile(1)
    track = 0
    midi.addTrackName(track, 0, "Track 1")
    midi.addTempo(track, 0, 120)  # 120 BPM
    time = 0

    for note in sequence:
        midi.addNote(track, 0, note, time, 1, 100)  # 1 is the duration of the note
        time += 1  # Each note is spaced by 1 time unit (quarter note)

    with open(f"{filename}.mid", "wb") as f:
        midi.writeFile(f)

# GUI: Load reference MIDI, run the genetic algorithm, and evolve the sequence
def run_gui():
    running = True
    clock = pygame.time.Clock()

    reference_sequence = []  # This will be populated when you load the reference file
    population = [generate_random_midi_sequence() for _ in range(POP_SIZE)]
    generation = 0

    while running:
        screen.fill(WHITE)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l:  # 'L' to load a reference MIDI file
                    filename = "reference.mid"  # Change to user input or file dialog in a real app
                    if os.path.exists(filename):
                        reference_sequence = generate_random_midi_sequence()  # Replace with actual parsing
                        print(f"Reference MIDI loaded: {filename}")
                    else:
                        print("No such file found.")
                elif event.key == pygame.K_e:  # 'E' to evolve the population
                    if reference_sequence:
                        population = evolve_population(population, reference_sequence)
                        generation += 1
                        print(f"Generation {generation} evolved")
                    else:
                        print("No reference loaded.")

                elif event.key == pygame.K_s:  # 'S' to save the best sequence as MIDI
                    if reference_sequence:
                        best_sequence = population[0]  # Take the top candidate
                        save_midi(best_sequence, f"evolved_generation_{generation}")
                        print(f"Saved evolved MIDI to 'evolved_generation_{generation}.mid'")

        # Display instructions
        text = font.render("Press L to Load Reference, E to Evolve, S to Save", True, BLACK)
        screen.blit(text, (20, 20))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

# Run the GUI
run_gui()
