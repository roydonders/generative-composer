import random
import os
import webbrowser
from music21 import stream, note, chord, midi
import http.server
import socketserver

# Path for generated MIDI files
midi_path = 'output.mid'


# Function to generate MIDI from a sequence
def generate_midi(sequence, filename='output.mid'):
    # Create a music21 stream
    s = stream.Stream()

    # Convert the sequence to music21 notes/chords
    for p in sequence:
        if p is None:
            s.append(note.Rest())
        elif isinstance(p, str):
            s.append(note.Note(p))
        elif isinstance(p, list):  # chords can be a list of pitches
            s.append(chord.Chord(p))

    # Write the stream as a MIDI file
    s.write('midi', fp=filename)


# Function to generate a random melody (or evolved MIDI sequence)
def generate_random_melody():
    # Example sequence (you can evolve this using your genetic algorithm)
    return ['C4', 'E4', 'G4', 'C5', None, 'B4']


# Genetic Algorithm to evolve the sequence
def evolve_population(population_size=4, mutation_rate=0.1, generations=5):
    population = [generate_random_melody() for _ in range(population_size)]

    for generation in range(generations):  # Number of generations
        print(f"Generation {generation + 1}:")

        # Select two parents
        parents = random.sample(population, 2)

        # Crossover - create new sequence from parents (simplified)
        crossover_point = len(parents[0]) // 2
        child = parents[0][:crossover_point] + parents[1][crossover_point:]

        # Mutation - change a random note
        if random.random() < mutation_rate:
            mutate_index = random.randint(0, len(child) - 1)
            child[mutate_index] = generate_random_melody()[0]  # Replace with a random note

        # Add child to population
        population.append(child)

        # Only keep the best population_size sequences
        population = sorted(population, key=lambda seq: evaluate_fitness(seq))[:population_size]

        # Generate MIDI for each candidate in the population
        for i, candidate in enumerate(population):
            midi_file = f'candidate_{i + 1}.mid'
            generate_midi(candidate, midi_file)

        print("MIDI files generated for each candidate.")

    return population


# Fitness function to evaluate the candidate sequences
def evaluate_fitness(sequence):
    # Simplified fitness function (to be improved)
    return len(sequence)  # Fitness based on length (can be replaced by a more complex function)


# Function to start the HTTP server and serve files
def start_http_server(port=8000):
    web_dir = os.path.abspath(".")
    Handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), Handler)
    print(f"Serving at http://localhost:{port}")

    # Automatically open the index.html file in the default web browser
    webbrowser.open(f'http://localhost:{port}/index.html')

    httpd.serve_forever()


# Start evolving the population and serving the MIDI files
if __name__ == "__main__":
    evolve_population()  # Evolve sequences
    start_http_server()  # Start the server to serve the MIDI files

