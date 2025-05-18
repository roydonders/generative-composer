# backend_logic.py
"""
Module: backend_logic.py

Purpose:
This module serves as the central backend processing unit for the MIDI Melody Evolver
application. It acts as an intermediary between the user interface (e.g., Gradio app)
and the core genetic algorithm components.

Key Responsibilities:
- MIDI File Processing: Loads and parses uploaded MIDI files to extract essential
  musical information such as the primary melody line (notes and rests), tempo,
  key signature, phrase structure, and total duration in seconds. This extracted
  information forms the "reference" for the genetic algorithm.
- Genetic Algorithm Orchestration: Initializes and invokes the genetic algorithm
  (from `ga_components.genetic_algorithm_core`) to evolve new melodies. It passes
  the reference melody's characteristics and other parameters (like tempo) to the GA.
- Music Stream Creation: Converts melodies (represented as lists of pitch/duration tuples)
  into `music21.stream.Stream` objects, which are a standard music21 representation
  that can be further processed or output.
- Audio Synthesis and Conversion: Handles the conversion of generated melodies into
  playable audio formats (MIDI files for download, MP3 files for in-app playback).
  This involves using libraries like `music21`, `pretty_midi`, `soundfile`, and `pydub`.

Design Philosophy:
The backend logic aims to encapsulate all complex processing tasks, providing clear
and simple function interfaces for the UI (Gradio app) to call. It ensures that
data is correctly formatted and passed between different components of the system.
Error handling and temporary file management for audio conversion are also handled here.
"""

# Standard library imports
import os
import tempfile  # For creating temporary files for MIDI and audio output
import random  # Currently not used directly in this file, but often useful in backends
from typing import List, Tuple, Optional, Dict, Any  # Set was unused

# Third-party library imports for music processing
from music21 import converter, stream, note, tempo, key  # Core music21 objects

# Third-party library imports for audio processing
from pydub import AudioSegment  # For MP3 conversion from WAV
import pretty_midi  # For alternative MIDI parsing and robust synthesis to WAV
import soundfile  # For writing NumPy audio data to WAV files

# Local application/library specific imports
# Import constants, type definitions, and GA components from the 'ga_components' package.
from ga_components.music_constants import (
    POPULATION_SIZE, NOTES_PER_PHRASE_FALLBACK, MelodySequence, ReferencePhraseInfo
)
from ga_components.music_utils import MusicUtils
from ga_components.melody_generator import MelodyGenerator
from ga_components.genetic_algorithm_core import GeneticAlgorithm


# Configuration for SoundFont (if music21's direct MIDI synthesis to audio was the primary method)
# This section is currently not critical as pretty_midi's basic synthesizer is used for MP3 generation.
# If direct synthesis via music21 (e.g., using FluidSynth) were re-enabled,
# ensuring music21's environment is correctly set up would be important.
# Example:
# from music21 import environment
# us = environment.UserSettings()
# us['midiPath'] = '/path/to/your/fluidsynth_or_timidity_executable'
# us['soundfontPath'] = '/path/to/your/soundfont.sf2'


def extract_melody_data_from_midi(
        file_path: str
) -> Tuple[
    List[Optional[str]],  # List of extracted pitch strings (or None for rests)
    List[float],  # List of corresponding durations in quarter notes
    int,  # Detected tempo in BPM (beats per minute)
    str,  # Detected key signature as a string (e.g., "C major")
    str,  # Original filename of the MIDI file
    Optional[str],  # Error message if loading/parsing fails, None otherwise
    Optional[List[ReferencePhraseInfo]],  # Analyzed phrase information from the MIDI
    float  # Total duration of the reference melody in seconds
]:
    """
    Loads a MIDI file and extracts its primary melody line, tempo, key signature,
    phrase structure, and calculates its total duration in seconds. This information
    serves as the reference for the genetic algorithm.

    Args:
        file_path (str): The absolute path to the MIDI file to be processed.

    Returns:
        A tuple containing all extracted musical data and metadata.
        If an error occurs, appropriate default values and an error message are returned.
    """
    default_total_duration_secs = 0.0  # Fallback duration if calculation fails.
    try:
        # Parse the MIDI file using music21's converter.
        score: stream.Score = converter.parse(file_path)

        # --- Key Signature Analysis ---
        # music21's analyze('key') attempts to determine the overall key of the score.
        detected_key_obj: key.Key = score.analyze('key')
        # Use the direct string representation of the music21.key.Key object.
        # This is typically well-formed (e.g., "C major", "a minor") and
        # should be reliably parsable by music21.key.Key() if needed later.
        key_name_for_return: str = str(detected_key_obj)

        # --- Melody Extraction ---
        # Assumes the primary melody is in the first part of the MIDI file.
        # For complex multi-part MIDIs, a more sophisticated part selection might be needed.
        # .flat provides a flattened view of all musical elements in a stream.
        target_stream_for_melody: stream.Voice = score.parts[0].flat if score.parts else score.flat

        melody_events: MelodySequence = []  # Stores (pitch_string_or_None, duration_float)
        for element in target_stream_for_melody.notesAndRests:  # Iterate over notes and rests
            if isinstance(element, note.Note):
                # Store note as (pitch_name_with_octave, quarter_length_duration)
                melody_events.append((element.nameWithOctave, float(element.quarterLength)))
            elif isinstance(element, note.Rest):
                # Store rest as (None_for_pitch, quarter_length_duration)
                melody_events.append((None, float(element.quarterLength)))
            # Chords are currently simplified by this extraction (only monophonic line).
            # Future enhancement could involve extracting top notes from chords or arpeggiating.

        if not melody_events:  # If no notes or rests were found
            return ([], [], 120, "C major", os.path.basename(file_path),
                    "No notes or rests found in the MIDI's main part.", None, default_total_duration_secs)

        # Separate pitches and rhythms into their own lists for easier use by the GA.
        reference_pitches: List[Optional[str]] = [event[0] for event in melody_events]
        reference_rhythms: List[float] = [event[1] for event in melody_events]  # Durations in quarter lengths

        # --- Tempo Detection ---
        # Use the first MetronomeMark found. Defaults to 120 BPM if none.
        tempo_bpm = 120  # Default tempo
        tempo_marks_in_score = score.flat.getElementsByClass(tempo.MetronomeMark)
        if tempo_marks_in_score and tempo_marks_in_score[0].number:  # Check if number attribute exists and is valid
            tempo_bpm = int(tempo_marks_in_score[0].number)

        # --- Calculate Total Duration in Seconds ---
        # Sum of all quarter length durations gives total duration in quarter notes.
        total_duration_in_quarter_notes = sum(reference_rhythms)
        # Convert to seconds: (total_qn / (beats_per_minute / 60_seconds_per_minute))
        # which simplifies to (total_qn / tempo_bpm) * 60.0
        reference_total_duration_secs = (total_duration_in_quarter_notes / tempo_bpm) * 60.0 if tempo_bpm > 0 else 0.0

        # --- Phrase Analysis of the Extracted Melody ---
        # Use MusicUtils to segment the melody into phrases and analyze their character.
        # This requires the detected music21.key.Key object for tonal context.
        analyzed_phrases: Optional[List[ReferencePhraseInfo]] = MusicUtils.analyze_reference_phrases(
            melody_events, detected_key_obj  # Pass the Key object directly, not its string representation
        )

        return (reference_pitches, reference_rhythms, tempo_bpm, key_name_for_return,
                os.path.basename(file_path), None, analyzed_phrases, reference_total_duration_secs)

    except Exception as e:
        error_message = f"Failed to load or parse MIDI '{os.path.basename(file_path)}': {e}"
        print(f"Error in extract_melody_data_from_midi: {error_message}")
        # Return default/empty values in case of an error.
        return ([], [], 120, "C major", os.path.basename(file_path), error_message, None, default_total_duration_secs)


def create_music21_stream(melody_tuples: MelodySequence,
                          tempo_val: int,
                          key_name_str: str) -> stream.Stream:
    """
    Creates a `music21.stream.Stream` object from a melody represented as a list of
    (pitch, duration) tuples. The stream includes tempo and key signature information,
    making it ready for MIDI file export or further music21 processing.

    Args:
        melody_tuples (MelodySequence): The melody to convert, as a list of
                                        (pitch_string_or_None, duration_float) tuples.
        tempo_val (int): The tempo for the stream in BPM.
        key_name_str (str): The key signature for the stream (e.g., "C major", "a minor").

    Returns:
        music21.stream.Stream: A Stream object representing the melody.
    """
    music_stream_obj = stream.Stream()  # Create an empty music21 stream

    # Add tempo marking to the beginning of the stream.
    music_stream_obj.append(tempo.MetronomeMark(number=int(tempo_val)))

    # Add key signature to the stream.
    # MusicUtils.get_key_object robustly parses the key string.
    key_obj_for_stream = MusicUtils.get_key_object(key_name_str)
    # music21 uses the number of sharps/flats for KeySignature.
    music_stream_obj.append(key.KeySignature(key_obj_for_stream.sharps))

    # Add notes and rests to the stream from the input melody_tuples.
    for pitch_str, duration_float in melody_tuples:
        try:
            m21_event: Any  # Can be music21.note.Note or music21.note.Rest
            if pitch_str is None:  # This tuple represents a rest.
                m21_event = note.Rest(quarterLength=float(duration_float))
            else:  # This tuple represents a note.
                m21_event = note.Note(pitch_str, quarterLength=float(duration_float))
            music_stream_obj.append(m21_event)
        except Exception as e_note_creation:
            # Log a warning if a specific note/rest cannot be created, but continue processing others.
            print(
                f"Warning (create_music21_stream): Could not create music21 event for ('{pitch_str}', {duration_float}). Error: {e_note_creation}")

    return music_stream_obj


def evolve_one_generation(
        current_population: List[MelodySequence],
        ref_pitches: List[Optional[str]],
        ref_rhythms: List[float],
        current_key_name: str,  # Key signature string (e.g., "c# minor")
        ref_phrases_info: Optional[List[ReferencePhraseInfo]],  # Analyzed phrases from reference
        tempo_bpm: int,  # Tempo of the reference MIDI, passed to GA for duration calculations
        ref_total_duration_secs: float  # Total duration of reference in seconds, passed to GA
) -> List[MelodySequence]:
    """
    Executes one generation of the genetic algorithm to evolve the population of melodies.
    It initializes the GA with reference data and current parameters, then runs its
    evolutionary cycle once.

    Args:
        current_population: The list of melodies from the previous generation.
                            If empty, the GA will initialize a new population.
        ref_pitches: Pitch sequence from the reference MIDI.
        ref_rhythms: Rhythm sequence (quarter lengths) from the reference MIDI.
        current_key_name: The musical key context for evolution.
        ref_phrases_info: Analyzed phrase information from the reference MIDI.
        tempo_bpm: The tempo of the reference context, used by the GA's fitness function.
        ref_total_duration_secs: The total duration of the reference melody in seconds.

    Returns:
        List[MelodySequence]: The new generation of melodies after evolution.
                              Returns the current population or a new random population if
                              critical inputs are missing or GA initialization fails.
    """
    # --- Input Validation and Fallbacks for GA Initialization ---
    # If no reference MIDI data (pitches/rhythms), the GA will rely more on intrinsic
    # musicality measures or basic random generation if starting from scratch.
    if not ref_pitches or not ref_rhythms:
        if not current_population:  # No reference AND no existing population to evolve from
            print(
                "Warning (evolve_one_generation): No reference MIDI and no current population. Generating a new initial population for evolution.")
            # Generate a default initial population.
            temp_melody_gen = MelodyGenerator(current_key_name)  # Uses the provided key
            # Target length can be arbitrary, e.g., based on NOTES_PER_PHRASE_FALLBACK.
            default_target_events = NOTES_PER_PHRASE_FALLBACK * 4  # Example: 4 phrases worth of events
            new_initial_pop = [
                mel for _ in range(POPULATION_SIZE)  # Generate POPULATION_SIZE individuals
                if (mel := temp_melody_gen.generate_melody(default_target_events, None))  # Ensure melody is not None
            ]
            return new_initial_pop
        # If no reference, but a population exists, the GA will proceed.
        # Its fitness function should be designed to handle cases with or without reference data.
        print(
            "Warning (evolve_one_generation): Evolving without strong reference MIDI data. Fitness will rely more on intrinsic heuristics.")

    # Determine target number of musical events for melodies generated by the GA.
    # If reference exists, aim to match its length. Otherwise, use a fallback.
    melody_target_event_count_for_ga = len(ref_pitches) if ref_pitches else NOTES_PER_PHRASE_FALLBACK * 4

    # --- Initialize and Run Genetic Algorithm ---
    ga_instance = GeneticAlgorithm(
        population_size=POPULATION_SIZE,  # From music_constants
        melody_target_event_count=melody_target_event_count_for_ga,
        key_name_str=current_key_name,
        tempo_bpm=tempo_bpm,  # Pass tempo for fitness calculations
        ref_pitches_list=ref_pitches,
        ref_rhythms_list=ref_rhythms,
        ref_phrases_data=ref_phrases_info,
        ref_total_duration_secs=ref_total_duration_secs  # Pass total duration for fitness
    )

    # Seed the GA's internal population with the `current_population` if provided and valid.
    # This allows for continuous evolution across multiple calls to `evolve_one_generation`.
    if current_population and len(current_population) == POPULATION_SIZE:
        # Basic validation: ensure current_population individuals are lists of (pitch, duration) tuples.
        is_valid_current_pop = all(
            isinstance(ind, list) and ind and all(isinstance(n, tuple) and len(n) == 2 for n in ind)
            for ind in current_population
        )
        if is_valid_current_pop:
            ga_instance.population = current_population  # Use the provided population.
        else:
            print(
                "Warning (evolve_one_generation): Provided current_population contains invalid individuals. GA will re-initialize its own population.")
            # If current_population is invalid, the GA instance will use its own `_initialize_population` method.
    elif not ga_instance.population and melody_target_event_count_for_ga > 0:
        # This case handles if the GA's own initialization (called in its __init__) somehow failed
        # and `current_population` was not provided or was invalid.
        print(
            "Error (evolve_one_generation): GA instance failed to initialize its population. Attempting fallback generation.")
        temp_melody_gen = MelodyGenerator(current_key_name)
        ref_phrase_durations_for_fallback_gen = [p['duration_beats'] for p in
                                                 ref_phrases_info] if ref_phrases_info else None
        fallback_pop = [
            mel for _ in range(POPULATION_SIZE)
            if (mel := temp_melody_gen.generate_melody(melody_target_event_count_for_ga,
                                                       ref_phrase_durations_for_fallback_gen))
        ]
        return fallback_pop

    # Run one generation of the genetic algorithm.
    new_generation_melodies = ga_instance.run_one_generation()
    return new_generation_melodies


def save_melody_to_midi_file(melody_tuples: MelodySequence,
                             tempo_val: int,
                             key_name: str,
                             base_filename: str = "evolved_melody") -> Optional[str]:
    """
    Saves a given melody (list of (pitch, duration) tuples) to a MIDI file
    in a temporary location. The caller is responsible for managing/deleting this file.

    Args:
        melody_tuples (MelodySequence): The melody to save.
        tempo_val (int): Tempo in BPM for the MIDI file.
        key_name (str): Key signature string for the MIDI file.
        base_filename (str): A base name for the temporary MIDI file (e.g., "evolved_melody_gen5").

    Returns:
        Optional[str]: The file path to the saved temporary MIDI file if successful, None otherwise.
    """
    try:
        # Create a music21 stream from the melody data using the helper function.
        music_stream_to_save = create_music21_stream(melody_tuples, tempo_val, key_name)

        # Create a named temporary file to store the MIDI output.
        # `delete=False` is crucial: music21.write closes the file pointer, and we need
        # the file to persist for the caller (e.g., Gradio for download).
        with tempfile.NamedTemporaryFile(suffix=".mid", prefix=base_filename + "_",
                                         delete=False) as tmp_midi_file_descriptor:
            midi_file_path = tmp_midi_file_descriptor.name
        # The file object is closed when exiting the 'with' block, but the file itself remains due to delete=False.

        # Write the music21 stream to the MIDI file.
        music_stream_to_save.write("midi", fp=midi_file_path)

        # Verify that the file was created and is not empty.
        if os.path.exists(midi_file_path) and os.path.getsize(midi_file_path) > 0:
            return midi_file_path
        else:
            print(f"Error (save_melody_to_midi_file): MIDI file '{midi_file_path}' was not created or is empty.")
            if os.path.exists(midi_file_path): os.unlink(midi_file_path)  # Clean up empty file
            return None
    except Exception as e:
        print(f"Error (save_melody_to_midi_file): Failed to save melody to MIDI file: {e}")
        return None


def convert_melody_to_mp3_file(
        melody_tuples: MelodySequence,
        tempo_val: int,
        key_name_str: str,
        base_filename: str = "evolved_audio"
) -> Optional[str]:
    """
    Converts a melody (list of (pitch, duration) tuples) to an MP3 audio file.
    This is a multi-step process:
    1. Create a music21 stream and save it as a temporary MIDI file.
    2. Load this temporary MIDI file using `pretty_midi`.
    3. Synthesize audio data (as a NumPy array) from the `pretty_midi` object using its
       basic built-in synthesizer. This avoids external dependencies like FluidSynth for simple playback.
    4. Save the synthesized audio data as a temporary WAV file using `soundfile`.
    5. Convert the temporary WAV file to MP3 format using `pydub`.
    All temporary files are cleaned up. The caller is responsible for the final MP3 file.

    Args:
        melody_tuples (MelodySequence): The melody to convert.
        tempo_val (int): Tempo in BPM for synthesis.
        key_name_str (str): Key signature string for context.
        base_filename (str): Base for the temporary MP3 filename.

    Returns:
        Optional[str]: Path to the temporary MP3 file if successful, None otherwise.
    """
    temp_midi_file_path: Optional[str] = None  # Path for intermediate MIDI
    temp_wav_file_path: Optional[str] = None  # Path for intermediate WAV
    final_mp3_path: Optional[str] = None  # Path for the final MP3 for Gradio/user

    try:
        # Step 1: Create music21 stream and save as temporary MIDI
        music_stream_for_conversion = create_music21_stream(melody_tuples, tempo_val, key_name_str)
        with tempfile.NamedTemporaryFile(suffix=".mid", prefix="temp_synth_midi_", delete=False) as tmp_midi:
            temp_midi_file_path = tmp_midi.name
        music_stream_for_conversion.write('midi', fp=temp_midi_file_path)

        # Validate MIDI file creation
        if not (temp_midi_file_path and os.path.exists(temp_midi_file_path) and os.path.getsize(
                temp_midi_file_path) > 0):
            raise RuntimeError(f"Temporary MIDI file creation failed or file is empty: {temp_midi_file_path}")

        # Step 2: Load MIDI with pretty_midi
        pretty_midi_object = pretty_midi.PrettyMIDI(temp_midi_file_path)

        # Step 3: Synthesize audio data using pretty_midi's basic synthesizer
        # fs=44100 Hz is a standard CD quality sampling rate.
        synthesized_audio_data = pretty_midi_object.synthesize(fs=44100)  # Returns a NumPy array

        # Step 4: Save synthesized audio as temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", prefix="temp_synth_wav_", delete=False) as tmp_wav:
            temp_wav_file_path = tmp_wav.name
        # `soundfile.write` expects (filepath, data, samplerate)
        soundfile.write(temp_wav_file_path, synthesized_audio_data, 44100)

        # Validate WAV file creation
        if not (temp_wav_file_path and os.path.exists(temp_wav_file_path) and os.path.getsize(temp_wav_file_path) > 0):
            raise RuntimeError(f"Temporary WAV file creation failed or file is empty: {temp_wav_file_path}")

        # Step 5: Convert WAV to MP3 using pydub
        audio_segment_from_wav = AudioSegment.from_wav(temp_wav_file_path)
        with tempfile.NamedTemporaryFile(suffix=".mp3", prefix=base_filename + "_", delete=False) as tmp_mp3:
            final_mp3_path = tmp_mp3.name
        # Export to MP3 format. Bitrate can be specified if needed (e.g., `bitrate="192k"`).
        audio_segment_from_wav.export(final_mp3_path, format="mp3")

        # Validate MP3 file creation
        if not (final_mp3_path and os.path.exists(final_mp3_path) and os.path.getsize(final_mp3_path) > 0):
            raise RuntimeError(f"Final MP3 file creation failed or file is empty: {final_mp3_path}")

        return final_mp3_path  # Success

    except Exception as e:
        print(
            f"ERROR (convert_melody_to_mp3_file) during audio conversion to MP3 (melody: {str(melody_tuples)[:70]}...): {type(e).__name__} - {e}")
        # Clean up any partially created MP3 file on error
        if final_mp3_path and os.path.exists(final_mp3_path):
            try:
                os.unlink(final_mp3_path)
            except OSError:
                pass  # Ignore if already deleted or permission issue
        return None  # Indicate failure
    finally:
        # Ensure all temporary intermediate files (MIDI, WAV) are cleaned up
        # regardless of success or failure of the overall conversion.
        if temp_wav_file_path and os.path.exists(temp_wav_file_path):
            try:
                os.unlink(temp_wav_file_path)
            except OSError as e_del_wav:
                print(f"Warning: Could not delete temp WAV file {temp_wav_file_path}: {e_del_wav}")
        if temp_midi_file_path and os.path.exists(temp_midi_file_path):
            try:
                os.unlink(temp_midi_file_path)
            except OSError as e_del_mid:
                print(f"Warning: Could not delete temp MIDI file {temp_midi_file_path}: {e_del_mid}")


if __name__ == '__main__':
    # This block allows testing backend functions directly if the script is run.
    # For example, one could load a test MIDI, evolve it, save output, etc.
    # This is useful for development and debugging isolated backend functionalities.
    print(f"--- backend_logic.py executed directly (for testing purposes) ---")
    # Example Usage (Illustrative - requires a test MIDI file named 'test.mid' in the same directory):
    # test_midi_file = "test.mid"
    # if os.path.exists(test_midi_file):
    #     print(f"Attempting to load test MIDI: {test_midi_file}")
    #     p, r, t_bpm, k_str, fname, err, phrases, total_s = extract_melody_data_from_midi(test_midi_file)
    #     if err:
    #         print(f"Test MIDI Load Error: {err}")
    #     else:
    #         print(f"Test MIDI Loaded: '{fname}', Key: {k_str}, Tempo: {t_bpm} BPM, Events: {len(p)}, Duration: {total_s:.2f}s")
    #         if phrases: print(f"Analyzed {len(phrases)} phrases from reference.")
    #
    #         # Test evolution (simplified, assuming data was loaded)
    #         if p: # If pitches were extracted
    #             print("\nTesting one generation of evolution...")
    #             evolved_generation = evolve_one_generation([], p, r, k_str, phrases, t_bpm, total_s)
    #             if evolved_generation and evolved_generation[0]:
    #                 first_evolved_melody = evolved_generation[0]
    #                 print(f"First melody from evolved generation ({len(first_evolved_melody)} events): {str(first_evolved_melody)[:60]}...")
    #
    #                 # Test saving the first evolved melody to MIDI
    #                 saved_midi_path = save_melody_to_midi_file(first_evolved_melody, t_bpm, k_str, "test_evolved_output")
    #                 if saved_midi_path:
    #                     print(f"Test evolved melody saved to MIDI: {saved_midi_path}")
    #                     # Test MP3 conversion from this saved MIDI (or directly from melody_tuples)
    #                     saved_mp3_path = convert_melody_to_mp3_file(first_evolved_melody, t_bpm, k_str, "test_evolved_audio")
    #                     if saved_mp3_path:
    #                         print(f"Test evolved audio saved to MP3: {saved_mp3_path}")
    #                         # Consider os.unlink(saved_mp3_path) here if you want to auto-cleanup test files.
    #                     # Consider os.unlink(saved_midi_path) for cleanup.
    # else:
    #     print("Test MIDI file 'test.mid' not found in current directory. Skipping direct execution tests.")
    print(f"--- End of backend_logic.py direct execution block ---")

# Informative print statement upon module loading when imported by another script (e.g., genetic_app.py).
# print(f"--- backend_logic.py (Comprehensive Comments, v. {__import__('datetime').datetime.now()}) loaded ---")
