import tkinter as tk
from tkinter import filedialog, messagebox
from music21 import converter, stream, note, tempo, key, midi, environment
import pygame
import random
import os
import tempfile
import webbrowser

# === Global State ===
reference_phrase_a_pitches = []    # Holds the original "A" phrase pitches
reference_phrase_a_rhythm = []      # Holds the original "A" phrase rhythm values
current_evolved = []             # Holds the current population of evolved refrains
tempo_bpm = 120                    # Default tempo in beats per minute
key_signature = "C"                # Default key signature
generations = 0                    # Counter for how many generations have been evolved
is_paused = False                  # Tracks playback state (pause/unpause)
refrain_structure = ["A", "A'", "B", "A''"] # Example refrain structure

# Setup music21 to show sheet music (assumes system default musicXML viewer)
us = environment.UserSettings()

# === Setup ===
pygame.init()
pygame.mixer.init()

# === Genetic Algorithm Functions ===
def generate_random_phrase(length):
    """Generate a random musical phrase."""
    scale_obj = key.Key(key_signature)
    scale_pitches = scale_obj.getPitches()
    weighted_notes = []

    for pitch in scale_pitches:
        scale_degree = scale_obj.getScaleDegreeFromPitch(pitch)
        weight = {7: 0.2, 4: 0.5}.get(scale_degree, 1.0)
        weighted_notes.extend([pitch.nameWithOctave] * int(weight * 10))

    rhythm = [random.choice([0.5, 1]) for _ in range(length)]
    phrase = [(random.choice(weighted_notes), dur) for dur in rhythm]
    return phrase

def generate_refrain_from_phrase(phrase_a, structure):
    """Generate a full refrain based on a core phrase and a structure."""
    refrain = []
    phrase_map = {"A": phrase_a}
    for part in structure:
        if part not in phrase_map:
            if part.startswith("A"):
                phrase_map[part] = mutate_phrase(phrase_map["A"], rate=0.1)
            elif part == "B":
                phrase_map[part] = generate_random_phrase(len(phrase_a))
            else:
                phrase_map[part] = phrase_map["A"] # Default to A if unknown
        refrain.extend(phrase_map[part])
    return refrain

def mutate_phrase(phrase, rate=0.1):
    """Randomly mutate notes and rhythms within a phrase."""
    scale_obj = key.Key(key_signature)
    scale_pitches = scale_obj.getPitches()
    weighted_notes = [pitch.nameWithOctave for pitch in scale_pitches for _ in range(5)] # Increased weight

    return [
        (
            n if random.random() > rate else random.choice(weighted_notes),
            d if random.random() > rate else random.choice([0.25, 0.5, 1])
        ) for n, d in phrase
    ]

def crossover_refrain(refrain1, refrain2, structure):
    """Perform crossover between two refrains, respecting the phrase structure."""
    new_refrain = []
    len_a = len(reference_phrase_a_pitches) # Assume all phrases are roughly the same length
    crossover_point = random.randint(1, len_a - 1)

    phrase_map1 = segment_refrain(refrain1, structure)
    phrase_map2 = segment_refrain(refrain2, structure)
    new_phrase_map = {}

    for part in structure:
        if random.random() < 0.5:
            new_phrase_map[part] = phrase_map1.get(part, [])
        else:
            new_phrase_map[part] = phrase_map2.get(part, [])

    # Basic crossover within corresponding phrase types
    for part in structure:
        if new_phrase_map.get(part) and len(new_phrase_map[part]) >= crossover_point:
            if part.startswith("A") or part == "B": # Apply crossover to A and B
                phrase1 = phrase_map1.get(part, [])
                phrase2 = phrase_map2.get(part, [])
                if len(phrase1) >= crossover_point and len(phrase2) >= crossover_point:
                    new_phrase = phrase1[:crossover_point] + phrase2[crossover_point:]
                    new_phrase_map[part] = new_phrase
                else:
                    new_phrase_map[part] = phrase1 if len(phrase1) >= len(phrase2) else phrase2
            new_refrain.extend(new_phrase_map[part])
        elif new_phrase_map.get(part):
            new_refrain.extend(new_phrase_map[part])

    return new_refrain

def segment_refrain(refrain, structure):
    """Segment a full refrain into its constituent phrases based on the structure."""
    phrase_map = {}
    index = 0
    phrase_length = len(reference_phrase_a_pitches) # Assume roughly equal length
    for part in structure:
        phrase_map[part] = refrain[index : index + phrase_length]
        index += phrase_length
    return phrase_map

def fitness(refrain):
    """Score a refrain based on how well its 'A' phrases match the reference 'A'."""
    score = 0
    refrain_segments = segment_refrain(refrain, refrain_structure)

    for part, evolved_phrase in refrain_segments.items():
        if part.startswith("A") and reference_phrase_a_pitches:
            for i, ((p, d), rp, rd) in enumerate(zip(evolved_phrase, reference_phrase_a_pitches, reference_phrase_a_rhythm)):
                phrase_weight = 2 if i % 4 in [0, 3] else 1  # Emphasize start/end of sub-phrase
                if p == rp:
                    score += phrase_weight
                if abs(d - rd) < 0.01:
                    score += phrase_weight
    return -score

def evolve_population():
    global current_evolved, generations
    if not reference_phrase_a_pitches:
        messagebox.showerror("Error", "No reference 'A' phrase loaded.")
        return

    if not current_evolved:
        initial_population = []
        initial_phrase_a = [(p, d) for p, d in zip(reference_phrase_a_pitches, reference_phrase_a_rhythm)]
        for _ in range(4):
            mutated_a = mutate_phrase(initial_phrase_a, rate=0.2)
            initial_population.append(generate_refrain_from_phrase(mutated_a, refrain_structure))
        current_evolved = initial_population

    offspring = []
    for i in range(len(current_evolved)):
        for j in range(i + 1, len(current_evolved)):
            child = crossover_refrain(current_evolved[i], current_evolved[j], refrain_structure)
            child = [mutate_phrase([note_dur], rate=0.05)[0] for note_dur in child] # Mutate each note/dur pair
            offspring.append(child)

    population = current_evolved + offspring
    population = sorted(population, key=lambda s: fitness(s))[:4]
    current_evolved = population

    generations += 1
    listbox.delete(0, tk.END)
    for i in range(len(current_evolved)):
        listbox.insert(tk.END, f"Gen {generations} - Refrain {i+1}")
    status_label.config(text=f"Evolved {generations} generations")

# === MIDI Utilities ===
def play_pause():
    """Toggle play/pause of the selected refrain."""
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
        refrain = current_evolved[index]

        s = stream.Stream()
        s.append(tempo.MetronomeMark(number=int(tempo_entry.get())))
        s.append(key.KeySignature(key.Key(key_signature).sharps))
        for p, d in refrain:
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
    """Save the selected evolved refrain to a MIDI file."""
    selection = listbox.curselection()
    if not selection:
        return
    index = selection[0]
    refrain = current_evolved[index]

    s = stream.Stream()
    s.append(tempo.MetronomeMark(number=int(tempo_entry.get())))
    s.append(key.KeySignature(key.Key(key_signature).sharps))
    for p, d in refrain:
        s.append(note.Note(p, quarterLength=d))

    file_path = filedialog.asksaveasfilename(defaultextension=".mid", filetypes=[("MIDI files", "*.mid")])
    if file_path:
        s.write("midi", fp=file_path)
        messagebox.showinfo("Saved", f"Saved as {file_path}")

def view_sheet_music():
    """Show sheet music for the selected refrain in system viewer."""
    selection = listbox.curselection()
    if not selection:
        return
    index = selection[0]
    refrain = current_evolved[index]

    s = stream.Stream()
    s.append(tempo.MetronomeMark(number=int(tempo_entry.get())))
    s.append(key.KeySignature(key.Key(key_signature).sharps))
    for p, d in refrain:
        s.append(note.Note(p, quarterLength=d))

    temp_file = tempfile.mktemp(suffix=".xml")
    s.write("musicxml", fp=temp_file)
    webbrowser.open(temp_file)

# === Load MIDI File and Extract Phrase A ===
def load_midi():
    """Load a MIDI file, attempt to identify a repeating 'A' phrase."""
    global reference_phrase_a_pitches, reference_phrase_a_rhythm, generations, current_evolved, key_signature, tempo_bpm

    file_path = filedialog.askopenfilename(filetypes=[("MIDI files", "*.mid")])
    if not file_path:
        return
    try:
        score = converter.parse(file_path)
        detected_key = score.analyze('key')
        key_signature = detected_key.tonic.name

        tempos = score.flat.getElementsByClass(tempo.MetronomeMark)
        if tempos and tempos[0].number:
            tempo_bpm = int(tempos[0].number)

        tempo_entry.delete(0, tk.END)
        tempo_entry.insert(0, str(tempo_bpm))
        generations = 0
        current_evolved = []
        listbox.delete(0, tk.END)

        # Attempt to find a repeating phrase (simplistic approach)
        notes = [n for n in score.parts[0].recurse().notes if isinstance(n, note.Note)]
        if not notes:
            messagebox.showerror("Error", "No notes found in the MIDI file.")
            return

        # Look for a sequence of 8 notes as a potential 'A' phrase (can be adjusted)
        phrase_length = 8
        best_match_start = -1
        max_self_similarity = -1

        for i in range(len(notes) - 2 * phrase_length + 1):
            phrase1_pitches = [n.nameWithOctave for n in notes[i : i + phrase_length]]
            phrase1_rhythm = [n.quarterLength for n in notes[i : i + phrase_length]]
            phrase2_pitches = [n.nameWithOctave for n in notes[i + phrase_length : i + 2 * phrase_length]]
            phrase2_rhythm = [n.quarterLength for n in notes[i + phrase_length : i + 2 * phrase_length]]

            pitch_similarity = sum(p1 == p2 for p1, p2 in zip(phrase1_pitches, phrase2_pitches))
            rhythm_similarity = sum(abs(r1 - r2) < 0.01 for r1, r2 in zip(phrase1_rhythm, phrase2_rhythm))
            total_similarity = pitch_similarity + rhythm_similarity

            if total_similarity > max_self_similarity:
                max_self_similarity = total_similarity
                best_match_start = i

        if best_match_start != -1:
            reference_phrase_a_pitches = [n.nameWithOctave for n in notes[best_match_start : best_match_start + phrase_length]]
            reference_phrase_a_rhythm = [n.quarterLength for n in notes[best_match_start : best_match_start + phrase_length]]
            status_label.config(text=f"Loaded {os.path.basename(file_path)} in {key_signature}. Found potential 'A' phrase ({len(reference_phrase_a_pitches)} notes).")
        else:
            # If no clear repeating phrase is found, take the first few notes as a starting point
            reference_phrase_a_pitches = [n.nameWithOctave for n in notes[:phrase_length]]
            reference_phrase_a_rhythm = [n.quarterLength for n in notes[:phrase_length]]
            status_label.config(text=f"Loaded {os.path.basename(file_path)} in {key_signature}. No clear repeating phrase found, using first {phrase_length} notes as 'A'.")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load MIDI: {e}")

# === GUI Setup ===
root = tk.Tk()
root.title("🎼 Refrain Evolver")
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

tk.Button(frame, text="🧬 Evolve Refrains", font=("Arial", 12), command=evolve_population).pack(fill="x", pady=6)

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
