# backend_logic.py
"""
Module: backend_logic.py

Purpose:
This module serves as the backend engine for the MIDI Evolver application.
It orchestrates the core functionalities, including:
- Loading and parsing MIDI files to extract musical information (melody, tempo, key).
- Interfacing with the genetic algorithm components (MelodyGenerator, GeneticAlgorithm)
  to initialize and evolve populations of melodies.
- Converting generated melodies into music21 streams.
- Synthesizing melodies into audio formats (MIDI, MP3) for playback and download.

It acts as a bridge between the user interface (e.g., Gradio app) and the
underlying music generation and processing logic.
"""

import os
import tempfile  # For creating temporary files for MIDI and audio
import random
from typing import List, Tuple, Optional, Dict, Any  # Set was unused

# Music processing libraries
from music21 import converter, stream, note, tempo, key  # midi module from music21 was unused

# Audio processing libraries
from pydub import AudioSegment  # For MP3 conversion
import pretty_midi  # For MIDI parsing and alternative synthesis
import soundfile  # For writing WAV files from synthesized audio
# import numpy # Imported but not directly used in this file's current logic.

# Import components from the Genetic Algorithm package (ga_components)
from ga_components.music_constants import (
    POPULATION_SIZE, NOTES_PER_PHRASE_FALLBACK, MelodySequence, ReferencePhraseInfo,
)
from ga_components.music_utils import MusicUtils
from ga_components.melody_generator import MelodyGenerator
from ga_components.genetic_algorithm_core import GeneticAlgorithm


def extract_melody_data_from_midi(
        file_path: str
) -> Tuple[List[Optional[str]], List[float], int, str, str, Optional[str], Optional[List[ReferencePhraseInfo]]]:
    """
    Loads a MIDI file, extracts the primary melody line (notes and rests),
    tempo, key signature, and analyzes its phrase structure.

    Args:
        file_path (str): The path to the MIDI file.

    Returns:
        Tuple containing:
            - List[Optional[str]]: Extracted pitches (e.g., "C4", None for rests).
            - List[float]: Corresponding durations in quarter notes.
            - int: Detected tempo in BPM (beats per minute).
            - str: Detected key signature (e.g., "C major", "a minor").
            - str: Original filename of the MIDI file.
            - Optional[str]: An error message if loading or parsing fails, None otherwise.
            - Optional[List[ReferencePhraseInfo]]: Analyzed phrase information from the MIDI.
                                                 None if analysis fails or no notes found.
    """
    try:
        score = converter.parse(file_path)

        # --- Key Signature Analysis ---
        detected_key_obj: key.Key = score.analyze('key')

        # Use the direct string representation of the detected music21.key.Key object.
        # This is typically well-formed (e.g., "C major", "a minor", "c# minor")
        # and should be reliably parsable by music21.key.Key() if needed later.
        key_name_for_return = str(detected_key_obj)

        # --- Melody Extraction ---
        target_stream = score.parts[0].flat if score.parts else score.flat
        melody_events: MelodySequence = []
        for element in target_stream.notesAndRests:
            if isinstance(element, note.Note):
                melody_events.append((element.nameWithOctave, float(element.quarterLength)))
            elif isinstance(element, note.Rest):
                melody_events.append((None, float(element.quarterLength)))

        if not melody_events:
            return ([], [], 120, "C major", os.path.basename(file_path),
                    "No notes or rests found in the MIDI's main part.", None)

        reference_pitches: List[Optional[str]] = [event[0] for event in melody_events]
        reference_rhythms: List[float] = [event[1] for event in melody_events]

        # --- Tempo Detection ---
        tempo_bpm = 120
        tempo_marks = score.flat.getElementsByClass(tempo.MetronomeMark)
        if tempo_marks and tempo_marks[0].number:
            tempo_bpm = int(tempo_marks[0].number)

        # --- Phrase Analysis of the extracted melody ---
        # Pass the actual detected_key_obj for analysis, not its string representation,
        # as MusicUtils.analyze_reference_phrases expects a music21.key.Key object.
        analyzed_phrases: Optional[List[ReferencePhraseInfo]] = MusicUtils.analyze_reference_phrases(
            melody_events, detected_key_obj  # Pass the Key object directly
        )

        return (reference_pitches, reference_rhythms, tempo_bpm, key_name_for_return,  # Use the robust key string
                os.path.basename(file_path), None, analyzed_phrases)

    except Exception as e:
        error_message = f"Failed to load or parse MIDI '{os.path.basename(file_path)}': {e}"
        print(f"Error in extract_melody_data_from_midi: {error_message}")
        # Return default/empty values in case of an error, including a safe default key string.
        return ([], [], 120, "C major", os.path.basename(file_path), error_message, None)


def create_music21_stream(melody_tuples: MelodySequence,
                          tempo_val: int,
                          key_name_str: str) -> stream.Stream:
    """
    Creates a music21.stream.Stream object from a melody defined by a list of
    (pitch, duration) tuples. The stream includes tempo and key signature information.

    Args:
        melody_tuples: A list of (pitch_string_or_None, duration_float) tuples.
        tempo_val: The tempo for the stream in BPM.
        key_name_str: The key signature for the stream (e.g., "C major").

    Returns:
        A music21.stream.Stream object representing the melody.
    """
    music_stream = stream.Stream()
    music_stream.append(tempo.MetronomeMark(number=int(tempo_val)))

    # MusicUtils.get_key_object will parse the key_name_str
    key_obj = MusicUtils.get_key_object(key_name_str)
    music_stream.append(key.KeySignature(key_obj.sharps))

    for pitch_str, duration_float in melody_tuples:
        try:
            m21_event: Any
            if pitch_str is None:
                m21_event = note.Rest(quarterLength=float(duration_float))
            else:
                m21_event = note.Note(pitch_str, quarterLength=float(duration_float))
            music_stream.append(m21_event)
        except Exception as e_note:
            print(f"Warning: Could not create music21 event for ('{pitch_str}', {duration_float}). Error: {e_note}")
    return music_stream


def evolve_one_generation(
        current_population: List[MelodySequence],
        ref_pitches: List[Optional[str]],
        ref_rhythms: List[float],
        current_key_name: str,  # This string should now be robust (e.g., "c# minor")
        ref_phrases_info: Optional[List[ReferencePhraseInfo]]
) -> List[MelodySequence]:
    """
    Executes one generation of the genetic algorithm to evolve the population of melodies.
    (Docstring and logic mostly unchanged from previous version, as the fix is in the input `current_key_name`)
    """
    if not ref_pitches or not ref_rhythms:
        if not current_population:
            print("No reference MIDI and no current population. Generating initial population for evolution.")
            temp_melody_gen = MelodyGenerator(current_key_name)  # current_key_name is used here
            default_target_events = NOTES_PER_PHRASE_FALLBACK * 4
            # Generate valid melodies
            new_pop = []
            for _ in range(POPULATION_SIZE):
                mel = temp_melody_gen.generate_melody(default_target_events, None)
                if mel: new_pop.append(mel)
            return new_pop
        print("Warning: Evolving without strong reference. Fitness will rely on intrinsic heuristics.")

    melody_target_event_count = len(ref_pitches) if ref_pitches else NOTES_PER_PHRASE_FALLBACK * 4

    ga_instance = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        melody_target_event_count=melody_target_event_count,
        key_name_str=current_key_name,  # This is passed to GA and then to MelodyGenerator
        ref_pitches_list=ref_pitches,
        ref_rhythms_list=ref_rhythms,
        ref_phrases_data=ref_phrases_info
    )

    if current_population and len(current_population) == POPULATION_SIZE:
        is_valid_population = all(
            isinstance(ind, list) and ind and all(isinstance(n, tuple) and len(n) == 2 for n in ind)
            for ind in current_population
        )
        if is_valid_population:
            ga_instance.population = current_population
        else:
            print(
                "Warning: Provided current_population contains invalid individuals. GA re-initializing its population.")
    elif not ga_instance.population and melody_target_event_count > 0:
        print("Error: GA failed to initialize its population. Attempting fallback generation.")
        temp_melody_gen = MelodyGenerator(current_key_name)
        ref_phrase_durations_for_gen = [p['duration_beats'] for p in ref_phrases_info] if ref_phrases_info else None
        # Generate valid melodies
        new_pop_fallback = []
        for _ in range(POPULATION_SIZE):
            mel = temp_melody_gen.generate_melody(melody_target_event_count, ref_phrase_durations_for_gen)
            if mel: new_pop_fallback.append(mel)
        return new_pop_fallback

    new_generation = ga_instance.run_one_generation()
    return new_generation


def save_melody_to_midi_file(melody_tuples: MelodySequence,
                             tempo_val: int,
                             key_name: str,
                             base_filename: str = "evolved_melody") -> Optional[str]:
    """
    Saves a given melody to a MIDI file in a temporary location.
    (Docstring and logic mostly unchanged)
    """
    try:
        music_stream = create_music21_stream(melody_tuples, tempo_val, key_name)
        with tempfile.NamedTemporaryFile(suffix=".mid", prefix=base_filename + "_", delete=False) as tmp_midi_fp_obj:
            midi_file_path = tmp_midi_fp_obj.name
        music_stream.write("midi", fp=midi_file_path)
        if os.path.exists(midi_file_path) and os.path.getsize(midi_file_path) > 0:
            return midi_file_path
        else:
            print(f"[SAVE_MIDI] Error: MIDI file '{midi_file_path}' was not created or is empty.")
            return None
    except Exception as e:
        print(f"[SAVE_MIDI] Error saving melody to MIDI file: {e}")
        return None


def convert_melody_to_mp3_file(
        melody_tuples: MelodySequence,
        tempo_val: int,
        key_name_str: str,
        base_filename: str = "evolved_audio"
) -> Optional[str]:
    """
    Converts a melody to an MP3 audio file.
    (Docstring and logic mostly unchanged)
    """
    temp_midi_file_path: Optional[str] = None
    temp_wav_file_path: Optional[str] = None
    gradio_mp3_path: Optional[str] = None

    try:
        music_stream = create_music21_stream(melody_tuples, tempo_val, key_name_str)
        with tempfile.NamedTemporaryFile(suffix=".mid", prefix="pm_midi_", delete=False) as tmp_midi:
            temp_midi_file_path = tmp_midi.name
        music_stream.write('midi', fp=temp_midi_file_path)

        if not (temp_midi_file_path and os.path.exists(temp_midi_file_path) and os.path.getsize(
                temp_midi_file_path) > 0):
            raise RuntimeError(f"Temporary MIDI file creation failed or file is empty: {temp_midi_file_path}")

        pm_object = pretty_midi.PrettyMIDI(temp_midi_file_path)
        audio_data = pm_object.synthesize(fs=44100)

        with tempfile.NamedTemporaryFile(suffix=".wav", prefix="pm_wav_", delete=False) as tmp_wav:
            temp_wav_file_path = tmp_wav.name
        soundfile.write(temp_wav_file_path, audio_data, 44100)

        if not (temp_wav_file_path and os.path.exists(temp_wav_file_path) and os.path.getsize(temp_wav_file_path) > 0):
            raise RuntimeError(f"Temporary WAV file creation failed or file is empty: {temp_wav_file_path}")

        audio_segment = AudioSegment.from_wav(temp_wav_file_path)
        with tempfile.NamedTemporaryFile(suffix=".mp3", prefix=base_filename + "_", delete=False) as tmp_mp3:
            gradio_mp3_path = tmp_mp3.name
        audio_segment.export(gradio_mp3_path, format="mp3")

        if not (gradio_mp3_path and os.path.exists(gradio_mp3_path) and os.path.getsize(gradio_mp3_path) > 0):
            raise RuntimeError(f"MP3 file creation failed or file is empty: {gradio_mp3_path}")
        return gradio_mp3_path

    except Exception as e:
        print(f"ERROR during audio conversion to MP3 (melody: {str(melody_tuples)[:50]}...): {e}")
        if gradio_mp3_path and os.path.exists(gradio_mp3_path):
            try:
                os.unlink(gradio_mp3_path)
            except OSError:
                pass
        return None
    finally:
        if temp_wav_file_path and os.path.exists(temp_wav_file_path):
            try:
                os.unlink(temp_wav_file_path)
            except OSError:
                pass
        if temp_midi_file_path and os.path.exists(temp_midi_file_path):
            try:
                os.unlink(temp_midi_file_path)
            except OSError:
                pass


if __name__ == '__main__':
    print(f"--- backend_logic.py executed directly (for testing) ---")
    # Add test cases here if desired
    print(f"--- End of backend_logic.py direct execution block ---")

# print(f"--- backend_logic.py (version {__import__('datetime').datetime.now()}) loaded ---")
