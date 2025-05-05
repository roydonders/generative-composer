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
phrase_lengths = []            # Holds the length of each phrase (number of notes)
phrase_starts = []             # Scale degree of the first note in each phrase
phrase_ends = []               # Scale degree of the last note in each phrase
rests = []                     # Holds the rests (whether they occur or not, and their lengths)


# Setup music21 to show sheet music (assumes system default musicXML viewer)
us = environment.UserSettings()

# === Setup ===
pygame.init()
pygame.mixer.init()


# === Utility: Phrase-aware grouping ===
def group_into_phrases(notes, rhythm, phrase_length=4):
    """Group a flat list of notes into phrases of fixed length, preserving rests."""
    phrases = []
    current_phrase = []
    current_rhythm = []
    for note, rest in zip(notes, rhythm):
        current_phrase.append(note)
        current_rhythm.append(rest)
        if len(current_phrase) >= phrase_length:
            phrases.append((current_phrase, current_rhythm))
            current_phrase = []
            current_rhythm = []
    if current_phrase:
        phrases.append((current_phrase, current_rhythm))
    return phrases


# === Genetic Algorithm Functions ===
def learn_phrasing(midi_stream):
    """Extract the phrasing structure from the loaded MIDI stream."""
    global phrase_lengths, phrase_starts, phrase_ends, rests
    phrase_lengths = []
    phrase_starts = []
    phrase_ends = []
    rests = []

    # Get the notes and rests
    notes = []
    rhythms = []
    for element in midi_stream.flat.notes:
        if isinstance(element, note.Note):
            notes.append(element.nameWithOctave)
            rhythms.append(element.quarterLength)
        elif isinstance(element, note.Rest):
            notes.append('rest')
            rhythms.append(element.quarterLength)

    # Check if we have any notes and rests
    if not notes:
        print("Warning: No notes or rests found in the MIDI file.")
        return

    print("Extracted notes and rhythms:", notes, rhythms)

    phrases = group_into_phrases(notes, rhythms)

    # Check if any phrases were found
    if not phrases:
        print("Warning: No phrases were found.")
        return

    for phrase, rhythm in phrases:
        phrase_lengths.append(len(phrase))
        phrase_starts.append(phrase[0])  # The first note/scale degree of the phrase
        phrase_ends.append(phrase[-1])  # The last note/scale degree of the phrase
        rests.append(any(n == 'rest' for n in phrase))  # Does the phrase contain a rest?

    print("Learned phrase structures:")
    print("Phrase lengths:", phrase_lengths)
    print("Phrase starts:", phrase_starts)
    print("Phrase ends:", phrase_ends)
    print("Rests:", rests)



def generate_melody_based_on_phrasing(length):
    """Generate a melody based on the learned phrasing structure."""
    melody = []
    num_phrases = length // 4  # Approximate number of phrases
    for i in range(num_phrases):
        phrase_len = random.choice(phrase_lengths)  # Learn how long each phrase should be
        phrase_start = random.choice(phrase_starts)  # Learn the starting note of the phrase
        phrase_end = random.choice(phrase_ends)  # Learn the ending note of the phrase

        # Construct a phrase with note-duration pairs
        phrase = []
        for _ in range(phrase_len - 1):
            # Copy rhythm from the original reference structure
            rhythm = random.choice(reference_rhythm)
            phrase.append((random.choice([phrase_start, phrase_end]), rhythm))  # Use the learned rhythm

        # Ensure the phrase ends with the learned end note
        phrase.append((phrase_end, rhythm))

        # Insert rests randomly within the phrase, if appropriate
        if random.random() < 0.5:  # 50% chance to add a rest
            rest_index = random.randint(0, len(phrase) - 1)
            phrase.insert(rest_index, ('rest', rhythm))  # Add a rest with the same rhythm as the surrounding notes

        melody.extend(phrase)

    return melody[:length]




def crossover(seq1, seq2):
    point = 4 * random.randint(1, len(seq1) // 4 - 1)
    return seq1[:point] + seq2[point:]


def mutate(seq, rate=0.1):
    scale_obj = key.Key(key_signature)
    scale_pitches = scale_obj.getPitches()

    weighted_notes = []
    for pitch in scale_pitches:
        scale_degree = scale_obj.getScaleDegreeFromPitch(pitch)
        if scale_degree == 7:
            weight = 0.3
        elif scale_degree == 4:
            weight = 0.6
        else:
            weight = 1.0
        weighted_notes.extend([pitch.nameWithOctave] * int(weight * 10))

    mutated = []
    for n, d in seq:
        if n == 'rest':  # Handle rest case separately
            new_note = note.Rest(quarterLength=d) if random.random() > rate else note.Rest(quarterLength=random.choice([0.25, 0.5, 1]))
        else:
            new_note = n if random.random() > rate else random.choice(weighted_notes)
            new_note = note.Note(new_note, quarterLength=d)  # Ensure it's a note object

        mutated.append((new_note, d))  # Preserve rhythm by attaching it to the mutated note/rest

    return mutated


def fitness(sequence):
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

    population = current_evolved if current_evolved else [generate_melody_based_on_phrasing(len(reference_sequence)) for
                                                          _ in range(4)]

    offspring = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            child = crossover(population[i], population[j])
            child = mutate(child)
            offspring.append(child)

    population += offspring
    population = sorted(population, key=lambda s: fitness(s))[:4]
    current_evolved = population

    generations += 1
    listbox.delete(0, tk.END)
    for i in range(len(population)):
        listbox.insert(tk.END, f"Gen {generations} - Melody {i + 1}")
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
            if p == 'rest':  # Handle the rest case
                s.append(note.Rest(quarterLength=d))  # Append a rest with the correct duration
            else:  # Handle the note case
                s.append(note.Note(p, quarterLength=d))  # Append a note with the correct pitch and duration

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

        learn_phrasing(score)  # Call the learn_phrasing function here
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
