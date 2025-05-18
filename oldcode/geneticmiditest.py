import tkinter as tk
from tkinter import filedialog, messagebox
from music21 import converter, stream, note, tempo, key, midi, environment
import pygame
import random
import os
import tempfile
import webbrowser

# === Global State ===
reference_sequence = []        # Holds the original melody pitches
reference_rhythm = []          # Holds the original melody rhythm values
current_evolved = []           # Holds the current population of evolved melodies
tempo_bpm = 120                # Default tempo in beats per minute
key_signature = "C"            # Default key signature
generations = 0                # Counter for how many generations have been evolved
is_paused = False              # Tracks playback state (pause/unpause)

# Setup music21 to show sheet music (assumes system default musicXML viewer)
us = environment.UserSettings()

# === Setup ===
pygame.init()
pygame.mixer.init()

# === Genetic Algorithm Functions ===
def generate_random_melody(length, base_rhythm=None):
    """Generate a refrain-like melody using repeated and slightly varied phrases."""
    scale_obj = key.Key(key_signature)
    scale_pitches = scale_obj.getPitches()
    weighted_notes = []

    for pitch in scale_pitches:
        scale_degree = scale_obj.getScaleDegreeFromPitch(pitch)
        weight = {7: 0.2, 4: 0.5}.get(scale_degree, 1.0)
        weighted_notes.extend([pitch.nameWithOctave] * int(weight * 10))

    def gen_phrase(rhythm_pattern):
        return [(random.choice(weighted_notes), dur) for dur in rhythm_pattern]

    # Determine rhythm pattern: use reference if available
    if base_rhythm:
        base_rhythm = base_rhythm[:length]
    else:
        base_rhythm = [random.choice([0.5, 1]) for _ in range(length)]

    # Build refrain structure: A A' B A
    phrase_len = len(base_rhythm) // 4
    A = gen_phrase(base_rhythm[:phrase_len])
    A_prime = mutate(A, rate=0.1)
    B = gen_phrase(base_rhythm[phrase_len:2*phrase_len])
    A2 = mutate(A, rate=0.05)

    melody = A + A_prime + B + A2
    return melody

def generate_refrain_melody(length, base_rhythm=None, num_repeats=4):
    """Generate a refrain-based melody: A A' A'' A'''."""
    scale_obj = key.Key(key_signature)
    scale_pitches = scale_obj.getPitches()
    weighted_notes = []

    for pitch in scale_pitches:
        scale_degree = scale_obj.getScaleDegreeFromPitch(pitch)
        weight = {7: 0.2, 4: 0.5}.get(scale_degree, 1.0)
        weighted_notes.extend([pitch.nameWithOctave] * int(weight * 10))

    # Set rhythm (use provided or generate random)
    phrase_len = length // num_repeats
    if base_rhythm:
        base_rhythm = base_rhythm[:phrase_len]
    else:
        base_rhythm = [random.choice([0.5, 1]) for _ in range(phrase_len)]

    # Create main phrase (A)
    phrase_A = [(random.choice(weighted_notes), dur) for dur in base_rhythm]

    # Create slight variations of A
    def vary(phrase, mutation_rate):
        return [(note if random.random() > mutation_rate else random.choice(weighted_notes), dur)
                for note, dur in phrase]

    # Repeat with variations
    melody = []
    for i in range(num_repeats):
        variation = vary(phrase_A, mutation_rate=0.1 if i > 0 else 0)
        melody.extend(variation)

    return melody


def crossover(seq1, seq2):
    """Perform crossover by slicing at a random point between phrases."""
    point = 4 * random.randint(1, len(seq1) // 4 - 1)
    return seq1[:point] + seq2[point:]

def mutate(seq, rate=0.1):
    """Randomly mutate notes and rhythms with a low probability, respecting phrasing."""
    scale_obj = key.Key(key_signature)
    scale_pitches = scale_obj.getPitches()

    weighted_notes = []
    for pitch in scale_pitches:
        scale_degree = scale_obj.getScaleDegreeFromPitch(pitch)
        if scale_degree == 7:
            weight = 0.2
        elif scale_degree == 4:
            weight = 0.5
        else:
            weight = 1.0
        weighted_notes.extend([pitch.nameWithOctave] * int(weight * 10))

    # Mutate note or rhythm in phrased manner
    return [
        (
            n if random.random() > rate else random.choice(weighted_notes),
            d if random.random() > rate else random.choice([0.25, 0.5, 1])
        ) for n, d in seq
    ]

def fitness(sequence):
    """Score a melody based on how well it matches the reference in pitch/rhythm, emphasizing phrases."""
    score = 0
    for i, ((p, d), rp, rd) in enumerate(zip(sequence, reference_sequence, reference_rhythm)):
        phrase_weight = 2 if i % 4 in [0, 3] else 1  # Emphasize start/end of phrase
        if p == rp:
            score += phrase_weight
        if abs(d - rd) < 0.01:
            score += phrase_weight
    return -score

def evolve_population():
    global current_evolved, generations
    if not reference_sequence:
        messagebox.showerror("Error", "No reference loaded.")
        return

    base_rhythm = reference_rhythm[:64]  # Limit to 16 bars max (assuming 4/4 and 1 note per beat)

    if not current_evolved:
        current_evolved = [generate_refrain_melody(length=len(reference_sequence), base_rhythm=reference_rhythm) for _ in range(4)]

    offspring = []
    for i in range(len(current_evolved)):
        for j in range(i + 1, len(current_evolved)):
            child = crossover(current_evolved[i], current_evolved[j])
            child = mutate(child)
            offspring.append(child)

    population = current_evolved + offspring
    population = sorted(population, key=lambda s: fitness(s))[:4]
    current_evolved = population

    generations += 1
    listbox.delete(0, tk.END)
    for i in range(len(current_evolved)):
        listbox.insert(tk.END, f"Gen {generations} - Melody {i+1}")
    status_label.config(text=f"Evolved {generations} generations")


# === MIDI Utilities ===
def play_pause():
    """Toggle play/pause of the selected melody."""
    global is_paused
    if pygame.mixer.music.get_busy() and not is_paused:
        pygame.mixer.music.pause()
        is_paused = True
    elif is_paused:
        pygame.mixer.music.unpause()
        is_paused = False
    else:
        selection = listbox.curselection()
        if not selection:
            return
        index = selection[0]
        melody = current_evolved[index]

        s = stream.Stream()
        s.append(tempo.MetronomeMark(number=int(tempo_entry.get())))
        s.append(key.KeySignature(key.Key(key_signature).sharps))
        for p, d in melody:
            s.append(note.Note(p, quarterLength=d))

        file_path = f"temp_play.mid"
        s.write("midi", fp=file_path)
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        is_paused = False

def stop_playback():
    """Stop the MIDI playback."""
    pygame.mixer.music.stop()

def save_selected():
    """Save the selected evolved melody to a MIDI file."""
    selection = listbox.curselection()
    if not selection:
        return
    index = selection[0]
    melody = current_evolved[index]

    s = stream.Stream()
    s.append(tempo.MetronomeMark(number=int(tempo_entry.get())))
    s.append(key.KeySignature(key.Key(key_signature).sharps))
    for p, d in melody:
        s.append(note.Note(p, quarterLength=d))

    file_path = filedialog.asksaveasfilename(defaultextension=".mid", filetypes=[("MIDI files", "*.mid")])
    if file_path:
        s.write("midi", fp=file_path)
        messagebox.showinfo("Saved", f"Saved as {file_path}")

def view_sheet_music():
    """Show sheet music for the selected melody in system viewer."""
    selection = listbox.curselection()
    if not selection:
        return
    index = selection[0]
    melody = current_evolved[index]

    s = stream.Stream()
    s.append(tempo.MetronomeMark(number=int(tempo_entry.get())))
    s.append(key.KeySignature(key.Key(key_signature).sharps))
    for p, d in melody:
        s.append(note.Note(p, quarterLength=d))

    temp_file = tempfile.mktemp(suffix=".xml")
    s.write("musicxml", fp=temp_file)
    webbrowser.open(temp_file)

# === Load MIDI File ===
def load_midi():
    """Load a reference melody from MIDI file and extract tempo and key."""
    global reference_sequence, reference_rhythm, generations, current_evolved, key_signature, tempo_bpm

    file_path = filedialog.askopenfilename(filetypes=[("MIDI files", "*.mid")])
    if not file_path:
        return
    try:
        score = converter.parse(file_path)
        detected_key = score.analyze('key')
        key_signature = detected_key.tonic.name

        melody = [n for n in score.parts[0].recurse().notes if isinstance(n, note.Note)]
        reference_sequence = [n.nameWithOctave for n in melody]
        reference_rhythm = [n.quarterLength for n in melody]

        tempos = score.flat.getElementsByClass(tempo.MetronomeMark)
        if tempos and tempos[0].number:
            tempo_bpm = int(tempos[0].number)

        tempo_entry.delete(0, tk.END)
        tempo_entry.insert(0, str(tempo_bpm))
        generations = 0
        current_evolved = []
        listbox.delete(0, tk.END)
        status_label.config(text=f"Loaded {os.path.basename(file_path)} in {key_signature} ({len(reference_sequence)} notes)")
        melody = [n for n in score.parts[0].recurse().notes if isinstance(n, note.Note)]
        reference_sequence = [n.nameWithOctave for n in melody][:64]  # 16 bars @ 4 notes/bar
        reference_rhythm = [n.quarterLength for n in melody][:64]
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load MIDI: {e}")

# === GUI Setup ===
root = tk.Tk()
root.title("🎼 MIDI Evolver")
root.geometry("440x580")
root.configure(bg="#f0f0f0")

frame = tk.Frame(root, bg="#f0f0f0")
frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

tk.Button(frame, text="🎵 Load MIDI", font=("Arial", 12), command=load_midi).pack(fill="x", pady=4)

row = tk.Frame(frame, bg="#f0f0f0")
row.pack(fill="x", pady=2)

tk.Label(row, text="Tempo:", bg="#f0f0f0").pack(side="left")
tempo_entry = tk.Entry(row)
tempo_entry.insert(0, "120")
tempo_entry.pack(side="left")

tk.Button(frame, text="🧬 Evolve Melody", font=("Arial", 12), command=evolve_population).pack(fill="x", pady=6)

listbox = tk.Listbox(frame, height=10, font=("Courier", 10))
listbox.pack(fill="both", expand=True, pady=5)

row2 = tk.Frame(frame, bg="#f0f0f0")
row2.pack(fill="x", pady=5)
tk.Button(row2, text="⏯ Play/Pause", command=play_pause).pack(side="left", expand=True, fill="x", padx=2)
tk.Button(row2, text="⏹ Stop", command=stop_playback).pack(side="left", expand=True, fill="x", padx=2)
tk.Button(row2, text="💾 Save", command=save_selected).pack(side="left", expand=True, fill="x", padx=2)
tk.Button(row2, text="🎼 View Score", command=view_sheet_music).pack(side="left", expand=True, fill="x", padx=2)

status_label = tk.Label(root, text="No MIDI loaded.", bg="#f0f0f0")
status_label.pack(pady=4)

root.mainloop()
