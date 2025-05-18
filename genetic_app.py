# genetic_app.py
"""
Module: genetic_app.py (formerly app.py)

Purpose:
This module defines the Gradio user interface for the MIDI Evolver application.
It provides interactive components for users to:
- Upload a reference MIDI file.
- View extracted musical information (key, tempo).
- Trigger the evolutionary process to generate new melodies.
- Select, play back (as MP3), and download (as MIDI or MP3) evolved melodies.

It uses functions from `backend_logic.py` to handle the core operations and
manages the state of the application within the Gradio interface.
"""

import gradio as gr # For creating the web user interface
import os # For path operations, though less used directly here now
from typing import List, Tuple, Optional, Dict, Any, cast # `cast` can be useful for type hinting complex Gradio states

# Import backend logic functions that interface with the core evolutionary engine.
import backend_logic

# Import type definitions from the central music_constants module
# This ensures type consistency between frontend and backend.
from ga_components.music_constants import MelodySequence, ReferencePhraseInfo

# --- Gradio Application State Management ---
# Gradio `gr.State` objects are used to maintain data across interactions
# within a user's session. For example, storing the reference melody's features,
# the current population of evolved melodies, etc.

# --- Gradio Interface Callbacks ---
# These functions are triggered by user interactions with Gradio components
# (e.g., button clicks, file uploads). They call backend logic, update
# application state, and modify UI components.

def handle_load_midi(
    midi_file_obj: Optional[gr.File], # Gradio File object, can be None
    current_tempo_gr_val: int,    # Current tempo value from Gradio UI
    current_key_gr_val: str       # Current key signature string from Gradio UI
) -> Tuple[
    List[Optional[str]],          # state: reference_pitches_state
    List[float],                  # state: reference_rhythms_state
    List[MelodySequence],         # state: current_evolved_melodies_state (initial population)
    int,                          # ui: tempo_input (updated with detected tempo)
    str,                          # ui: key_display (updated with detected key)
    int,                          # state: generation_count_state (reset to 0)
    str,                          # ui: status_display (feedback message)
    gr.Dropdown,                  # ui: evolved_melody_selector (updated choices)
    Optional[str],                # ui: audio_player (cleared)
    str,                          # ui: generation_display (reset to "0")
    Optional[List[ReferencePhraseInfo]] # state: reference_phrases_info_state
]:
    """
    Callback function for the MIDI file upload button.
    Processes the uploaded MIDI file:
    1. Extracts melody, tempo, key, and phrase information using `backend_logic`.
    2. Generates an initial population of melodies based on the reference.
    3. Updates relevant Gradio UI components and application state.

    Args:
        midi_file_obj: The Gradio File object from the upload component.
        current_tempo_gr_val: Tempo value from the UI (used as fallback if detection fails).
        current_key_gr_val: Key signature from the UI (used as fallback).

    Returns:
        A tuple of values to update various Gradio components and state objects.
    """
    # Default return values in case of an early error or no file.
    error_return_tuple = (
        [], [], [], current_tempo_gr_val, current_key_gr_val, 0,
        "Error during MIDI load. Please try another file or check console.",
        gr.Dropdown(choices=[], value=None), None, "0", None
    )

    if midi_file_obj is None:
        return ([], [], [], current_tempo_gr_val, current_key_gr_val, 0,
                "No MIDI file provided. Please upload a MIDI file.",
                gr.Dropdown(choices=[], value=None), None, "0", None)

    # The Gradio File object has a 'name' attribute which is the temporary path to the uploaded file.
    file_path = midi_file_obj.name

    # Call backend logic to extract data from the MIDI file.
    (extracted_pitches, extracted_rhythms, detected_tempo, detected_key,
     filename, error_msg, ref_phrases_analysis) = backend_logic.extract_melody_data_from_midi(file_path)

    if error_msg:
        # If backend reported an error, display it to the user.
        status = f"Error loading '{filename}': {error_msg}"
        # Retain current key/tempo if detection failed but some info might be partially useful
        return ([], [], [], detected_tempo or current_tempo_gr_val, detected_key or current_key_gr_val,
                0, status, gr.Dropdown(choices=[], value=None), None, "0", None)

    if not extracted_pitches:
        # If MIDI loaded but no notes/rests were found.
        status = f"Loaded '{filename}', but no notes/rests were extracted. Cannot proceed with evolution."
        return ([], [], [], detected_tempo, detected_key, 0, status,
                gr.Dropdown(choices=[], value=None), None, "0", None)

    # --- Generate Initial Population ---
    # Based on the extracted reference melody, create an initial set of melodies.
    initial_population: List[MelodySequence] = []
    if extracted_pitches: # Ensure there's a reference to base generation on.
        # The MelodyGenerator from backend_logic will use the detected key.
        # The number of events in the reference melody guides the length of generated melodies.
        # ref_phrases_analysis (if available) can guide phrase structure.

        # Pass reference phrase durations to the backend evolution trigger,
        # which will then pass it to the GA and MelodyGenerator.
        # For direct initial population generation here, we simulate one evolution step essentially.
        initial_population = backend_logic.evolve_one_generation(
            current_population=[], # Start with no prior population
            ref_pitches=extracted_pitches,
            ref_rhythms=extracted_rhythms,
            current_key_name=detected_key,
            ref_phrases_info=ref_phrases_analysis
        )


    if not initial_population and extracted_pitches:
        # This case should be rare if evolve_one_generation has fallbacks.
        status = f"Loaded '{filename}', but failed to generate an initial population."
        # Return extracted data but empty population.
        return (extracted_pitches, extracted_rhythms, [], detected_tempo, detected_key, 0, status,
                gr.Dropdown(choices=[], value=None), None, "0", ref_phrases_analysis)

    # Prepare choices for the melody selection dropdown.
    dropdown_choices = [f"Initial Melody {i + 1}" for i in range(len(initial_population))]
    status_message = (f"Successfully loaded: {filename}\n"
                      f"Key: {detected_key}, Tempo: {detected_tempo} BPM, Events: {len(extracted_pitches)}.\n"
                      f"Initial population generated ({len(initial_population)} melodies).\n"
                      f"Reference phrases analyzed: {len(ref_phrases_analysis) if ref_phrases_analysis else 0}")
    current_generation_count = 0 # Reset generation counter on new MIDI load.

    return (
        extracted_pitches, extracted_rhythms, initial_population,
        detected_tempo, detected_key, current_generation_count,
        status_message,
        gr.Dropdown(choices=dropdown_choices, value=dropdown_choices[0] if dropdown_choices else None),
        None, # Clear audio player
        str(current_generation_count), # Update generation display
        ref_phrases_analysis # Store the analyzed phrase info from reference
    )


def handle_evolve_melodies(
    ref_pitches_state: List[Optional[str]],
    ref_rhythms_state: List[float],
    current_pop_state: List[MelodySequence],
    gen_count_state: int,
    key_sig_from_display: str,      # Key signature currently shown in UI
    tempo_from_input: int,          # Tempo currently shown in UI (not directly used for GA core, but for context)
    ref_phrases_info_state: Optional[List[ReferencePhraseInfo]] # Analyzed phrases from reference
) -> Tuple[
    List[MelodySequence],         # state: current_evolved_melodies_state (new population)
    int,                          # state: generation_count_state (incremented)
    str,                          # ui: status_display
    gr.Dropdown,                  # ui: evolved_melody_selector
    Optional[str],                # ui: audio_player (cleared)
    str                           # ui: generation_display
]:
    """
    Callback for the "Evolve New Melodies" button.
    Triggers one generation of the genetic algorithm using `backend_logic`.

    Args:
        ref_pitches_state: Stored pitch sequence from the reference MIDI.
        ref_rhythms_state: Stored rhythm sequence from the reference MIDI.
        current_pop_state: The current population of evolved melodies.
        gen_count_state: The current generation number.
        key_sig_from_display: The key signature from the UI.
        tempo_from_input: The tempo from the UI (mainly for context, GA uses key).
        ref_phrases_info_state: Stored analyzed phrase information from the reference.

    Returns:
        A tuple of values to update Gradio components and state.
    """
    current_key_for_evolution = key_sig_from_display
    if not ref_pitches_state or not ref_rhythms_state:
        # Cannot evolve if no reference MIDI has been loaded and processed.
        status = "Cannot evolve: No reference MIDI loaded or data is missing."
        # Keep current state for dropdown if it exists
        old_choices = [f"Gen {gen_count_state} - Melody {i + 1}" for i in range(len(current_pop_state))] if current_pop_state else []
        return (current_pop_state, gen_count_state, status,
                gr.Dropdown(choices=old_choices, value=old_choices[0] if old_choices else None),
                None, str(gen_count_state))

    # Call backend logic to run one generation of evolution.
    new_population = backend_logic.evolve_one_generation(
        current_pop_state, ref_pitches_state, ref_rhythms_state,
        current_key_for_evolution, ref_phrases_info_state
    )

    if not new_population:
        # Handle case where evolution fails to produce a valid new population.
        old_choices = [f"Gen {gen_count_state} - Melody {i + 1}" for i in range(len(current_pop_state))] if current_pop_state else []
        status = "Evolution failed to produce a new valid population. Current population retained. Check console for errors."
        return (current_pop_state, gen_count_state, status,
                gr.Dropdown(choices=old_choices, value=old_choices[0] if old_choices else None),
                None, str(gen_count_state))

    updated_gen_count = gen_count_state + 1
    dropdown_choices = [f"Gen {updated_gen_count} - Melody {i + 1}" for i in range(len(new_population))]
    status_message = f"Evolved to Generation {updated_gen_count}. Population size: {len(new_population)}."

    return (
        new_population, updated_gen_count, status_message,
        gr.Dropdown(choices=dropdown_choices, value=dropdown_choices[0] if dropdown_choices else None),
        None, # Clear audio player on new evolution
        str(updated_gen_count) # Update generation display
    )


def handle_play_selected_melody(
    selected_melody_choice_str: Optional[str], # String from the dropdown (e.g., "Gen 1 - Melody 5")
    evolved_melodies_list_state: List[MelodySequence],
    tempo_val_from_input: int,
    key_sig_from_display: str
) -> Tuple[Optional[str], str]: # (audio_player_path, status_message)
    """
    Callback when a melody is selected from the dropdown.
    Generates an MP3 for the selected melody and updates the audio player.

    Args:
        selected_melody_choice_str: The string identifying the chosen melody from the dropdown.
        evolved_melodies_list_state: The current list of all evolved melodies.
        tempo_val_from_input: Current tempo from UI for audio synthesis.
        key_sig_from_display: Current key signature from UI for audio synthesis.

    Returns:
        Tuple: (path_to_mp3_audio_file_or_None, status_message_string)
    """
    final_mp3_path: Optional[str] = None
    status_message = "Processing selection for playback..."

    if not selected_melody_choice_str or not evolved_melodies_list_state:
        status_message = "No melody selected or no melodies available to play."
        return final_mp3_path, status_message

    try:
        # Extract the index from the dropdown string (e.g., "Gen X - Melody Y" -> index Y-1)
        # This parsing needs to be robust to changes in dropdown label format.
        parts = selected_melody_choice_str.split("Melody ")
        if len(parts) < 2 or not parts[-1].strip().isdigit(): # Check if last part is a number
             raise ValueError("Invalid melody selection string format.")
        melody_index = int(parts[-1].strip()) - 1 # Convert to 0-based index
    except ValueError as e:
        status_message = f"Error identifying selected melody from '{selected_melody_choice_str}': {e}"
        return final_mp3_path, status_message

    if not (0 <= melody_index < len(evolved_melodies_list_state)):
        status_message = f"Selected melody index {melody_index + 1} is out of bounds for the current population of {len(evolved_melodies_list_state)} melodies."
        return final_mp3_path, status_message

    selected_melody_data = evolved_melodies_list_state[melody_index]

    try:
        current_tempo = int(tempo_val_from_input) # Ensure tempo is integer
    except ValueError:
        status_message = "Invalid tempo value. Using default 120 BPM for playback."
        current_tempo = 120 # Fallback tempo

    # Call backend to convert the selected melody to an MP3 file.
    final_mp3_path = backend_logic.convert_melody_to_mp3_file(
        selected_melody_data, current_tempo, key_sig_from_display,
        base_filename="playback_audio" # Temporary filename base
    )

    if final_mp3_path and os.path.exists(final_mp3_path):
        status_message = f"Audio ready for: {selected_melody_choice_str}"
    else:
        status_message = f"Failed to generate audio for {selected_melody_choice_str}. Check console for errors."
        final_mp3_path = None # Ensure path is None if generation failed

    return final_mp3_path, status_message


def handle_save_selected_file(
    file_type_to_save: str, # "MIDI" or "MP3"
    selected_melody_choice_str: Optional[str],
    evolved_melodies_list_state: List[MelodySequence],
    tempo_val_from_input: int,
    key_sig_from_display: str
) -> Tuple[Optional[str], str]: # (output_file_path_for_download_or_None, status_message)
    """
    Generic handler for saving the selected melody as either a MIDI or MP3 file.
    This function is called by specific MIDI/MP3 save button handlers.

    Args:
        file_type_to_save: String indicating "MIDI" or "MP3".
        selected_melody_choice_str: String from the melody selection dropdown.
        evolved_melodies_list_state: Current list of evolved melodies.
        tempo_val_from_input: Current tempo from UI.
        key_sig_from_display: Current key signature from UI.

    Returns:
        Tuple: (path_to_generated_file_for_download_or_None, status_message_string)
    """
    output_file_path: Optional[str] = None
    status_message = f"Processing request to save as {file_type_to_save}..."

    if not selected_melody_choice_str or not evolved_melodies_list_state:
        status_message = f"No melody selected for {file_type_to_save} download."
        return output_file_path, status_message

    try:
        parts = selected_melody_choice_str.split("Melody ")
        if len(parts) < 2 or not parts[-1].strip().isdigit():
            raise ValueError("Invalid melody selection string format.")
        melody_index = int(parts[-1].strip()) - 1
    except ValueError as e:
        status_message = f"Error identifying selected melody for {file_type_to_save} download: {e}"
        return output_file_path, status_message

    if not (0 <= melody_index < len(evolved_melodies_list_state)):
        status_message = f"Selected melody index {melody_index + 1} is out of bounds for {file_type_to_save} download."
        return output_file_path, status_message

    selected_melody_data = evolved_melodies_list_state[melody_index]

    try:
        current_tempo = int(tempo_val_from_input)
    except ValueError:
        status_message = f"Invalid tempo value. Using default 120 BPM for {file_type_to_save} file."
        current_tempo = 120


    # Sanitize parts of the filename
    safe_key_sig = key_sig_from_display.replace(" ", "_").replace("#", "sharp").replace("b", "flat")
    generation_info_part = selected_melody_choice_str.split(" - ")[0].replace(" ", "") if " - " in selected_melody_choice_str else "Initial"
    melody_num_part = selected_melody_choice_str.split("Melody ")[-1].strip()
    suggested_filename_base = f"{generation_info_part}_Melody{melody_num_part}_{safe_key_sig}_{current_tempo}bpm"

    if file_type_to_save == "MIDI":
        output_file_path = backend_logic.save_melody_to_midi_file(
            selected_melody_data, current_tempo, key_sig_from_display,
            base_filename=suggested_filename_base
        )
    elif file_type_to_save == "MP3":
        output_file_path = backend_logic.convert_melody_to_mp3_file(
            selected_melody_data, current_tempo, key_sig_from_display,
            base_filename=suggested_filename_base + "_audio" # Add suffix for MP3
        )
    else:
        status_message = f"Unsupported file type for saving: {file_type_to_save}"
        return None, status_message

    if output_file_path and os.path.exists(output_file_path):
        status_message = f"{file_type_to_save} file for '{selected_melody_choice_str}' prepared for download."
    else:
        status_message = f"Failed to generate {file_type_to_save} file for '{selected_melody_choice_str}'. Check console."
        output_file_path = None

    return output_file_path, status_message


# --- Gradio UI Definition ---
# `gr.Blocks` allows for more complex and custom layouts compared to `gr.Interface`.
with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue)) as demo:
    gr.Markdown("# 🎵 MIDI Melody Evolver 🎶")
    gr.Markdown(
        "Upload a reference MIDI file, then use genetic algorithms to evolve new melodies based on its characteristics! "
        "You can play back the evolved melodies and download them as MIDI or MP3 files."
    )

    # --- Application State Variables ---
    # These `gr.State` objects hold data that persists across user interactions
    # but are not directly visible as UI components.
    reference_pitches_state = gr.State([]) # Stores pitches from the reference MIDI
    reference_rhythms_state = gr.State([]) # Stores rhythms from the reference MIDI
    current_evolved_melodies_state = gr.State([]) # Stores the current population of evolved melodies
    generation_count_state = gr.State(0) # Tracks the number of generations evolved
    reference_phrases_info_state = gr.State(None) # Stores analyzed phrase data from reference MIDI

    # --- UI Layout ---
    with gr.Row(): # Main row for input controls and status
        with gr.Column(scale=1): # Column for MIDI loading and parameters
            gr.Markdown("### 1. Load Reference MIDI")
            load_midi_button = gr.UploadButton(
                "📁 Load Reference MIDI File",
                file_types=[".mid", ".midi"], # Specify allowed file types
                # file_count="single" # Ensure only one file can be uploaded
            )
            tempo_input = gr.Number(label="Tempo (BPM)", value=120, interactive=True, step=1, precision=0)
            key_display = gr.Textbox(label="Detected Key Signature", value="C major", interactive=False)

        with gr.Column(scale=2): # Column for status messages and generation info
            gr.Markdown("### Application Status")
            status_display = gr.Textbox(
                label="Log & Status Messages",
                value="Please load a MIDI file to begin the evolutionary process.",
                interactive=False,
                lines=5, # Allow multiple lines for detailed messages
                max_lines=10
            )
            generation_display = gr.Textbox(label="Current Generation", value="0", interactive=False)

    gr.Markdown("---") # Separator

    gr.Markdown("### 2. Evolve Melodies")
    evolve_button = gr.Button("🧬 Evolve New Melodies (Next Generation)", variant="primary")

    gr.Markdown("---") # Separator

    gr.Markdown("### 3. Manage and Export Evolved Melodies")
    evolved_melody_selector = gr.Dropdown(
        label="Select Evolved Melody for Playback/Download",
        choices=[], # Initially empty, populated after MIDI load/evolution
        interactive=True
    )
    audio_player = gr.Audio(label="Melody Playback (MP3)", type="filepath", interactive=False)

    with gr.Row(): # Row for download buttons
        save_midi_button = gr.Button("💾 Save Selected as MIDI")
        save_mp3_button = gr.Button("💾 Save Selected as MP3")

    # Hidden File components for triggering downloads.
    # When these components receive a file path, Gradio enables download for that file.
    download_midi_output_file = gr.File(label="Download MIDI File", interactive=False) # Removed visible=False, Gradio handles visibility
    download_mp3_output_file = gr.File(label="Download MP3 File", interactive=False)


    # --- Event Handling: Connecting UI components to callback functions ---

    # When a MIDI file is uploaded via `load_midi_button`:
    load_midi_button.upload(
        fn=handle_load_midi,
        inputs=[load_midi_button, tempo_input, key_display],
        outputs=[
            reference_pitches_state, reference_rhythms_state,
            current_evolved_melodies_state, tempo_input, key_display,
            generation_count_state, status_display, evolved_melody_selector,
            audio_player, generation_display, reference_phrases_info_state
        ]
    )

    # When the `evolve_button` is clicked:
    evolve_button.click(
        fn=handle_evolve_melodies,
        inputs=[
            reference_pitches_state, reference_rhythms_state,
            current_evolved_melodies_state, generation_count_state,
            key_display, tempo_input, reference_phrases_info_state
        ],
        outputs=[
            current_evolved_melodies_state, generation_count_state, status_display,
            evolved_melody_selector, audio_player, generation_display
        ]
    )

    # When the selection in `evolved_melody_selector` (dropdown) changes:
    evolved_melody_selector.change(
        fn=handle_play_selected_melody,
        inputs=[evolved_melody_selector, current_evolved_melodies_state, tempo_input, key_display],
        outputs=[audio_player, status_display]
    )

    # When `save_midi_button` is clicked:
    save_midi_button.click(
        fn=lambda sel, pop, tempo, key: handle_save_selected_file("MIDI", sel, pop, tempo, key), # Lambda to pass file_type
        inputs=[evolved_melody_selector, current_evolved_melodies_state, tempo_input, key_display],
        outputs=[download_midi_output_file, status_display]
    )

    # When `save_mp3_button` is clicked:
    save_mp3_button.click(
        fn=lambda sel, pop, tempo, key: handle_save_selected_file("MP3", sel, pop, tempo, key), # Lambda to pass file_type
        inputs=[evolved_melody_selector, current_evolved_melodies_state, tempo_input, key_display],
        outputs=[download_mp3_output_file, status_display]
    )


if __name__ == "__main__":
    # Launch the Gradio application.
    # `debug=True` enables more detailed error messages during development.
    # `inbrowser=True` attempts to open the app in a new browser tab automatically.
    # `share=True` would generate a public link if you need to share it (requires internet).
    print("Launching Gradio MIDI Melody Evolver application...")
    demo.launch(debug=True, inbrowser=True)