# genetic_app.py
"""
Module: genetic_app.py

Purpose:
This module defines the Gradio-based web user interface for the MIDI Melody Evolver
application. It provides interactive components that allow users to:
- Upload a reference MIDI file to serve as the basis for evolution.
- View key musical information extracted from the reference (e.g., key, tempo, duration).
- Trigger the genetic algorithm to evolve new melodies based on the reference.
- Browse, select, and listen to (via MP3 playback) the evolved melodies.
- Download selected evolved melodies as either MIDI or MP3 files.

It interfaces with `backend_logic.py` to perform all core operations like MIDI
processing, melody evolution, and audio conversion. The state of the application
(e.g., reference melody data, current population of evolved melodies) is managed
using Gradio's `gr.State` components.

Design Philosophy:
The UI aims to be intuitive and provide clear feedback to the user. Callbacks
are linked to UI events (button clicks, file uploads, dropdown changes) and orchestrate
the flow of data between the frontend, backend, and application state. Error handling
and status messages are provided to keep the user informed.
"""

# Standard library imports
import os  # For path operations, e.g., checking if generated files exist.
from typing import List, Tuple, Optional, Dict, Any, cast  # For type hinting.

# Third-party library imports
import gradio as gr  # The core library for building the web UI.

# Local application/library specific imports
import backend_logic  # Handles all core MIDI processing and GA logic.
# Import type definitions for consistency with backend data structures.
from ga_components.music_constants import MelodySequence, ReferencePhraseInfo


# --- Gradio Interface Callback Functions ---
# These functions are triggered by user interactions with the Gradio UI components.
# They typically:
#   1. Receive input from UI components and/or application state.
#   2. Call functions in `backend_logic.py` to perform operations.
#   3. Update application state (`gr.State` objects).
#   4. Return values to update various Gradio UI components (e.g., text displays, dropdowns).

def handle_load_midi(
        midi_file_obj: Optional[gr.File],  # Gradio File object from the upload component; can be None.
        current_tempo_gr_val: int,  # Current tempo value from the Gradio UI's tempo input field.
        current_key_gr_val: str  # Current key signature string from the Gradio UI's key display.
) -> Tuple[  # Defines the structure of the tuple returned to update multiple Gradio outputs.
    List[Optional[str]],  # Output for: reference_pitches_state
    List[float],  # Output for: reference_rhythms_state
    List[MelodySequence],  # Output for: current_evolved_melodies_state (initial population)
    int,  # Output for: tempo_input (updated with detected tempo)
    str,  # Output for: key_display (updated with detected key)
    int,  # Output for: generation_count_state (reset to 0)
    str,  # Output for: status_display (feedback message to user)
    gr.Dropdown,  # Output for: evolved_melody_selector (updated choices)
    Optional[str],  # Output for: audio_player (cleared, as no melody is selected yet)
    str,  # Output for: generation_display (reset to "0")
    Optional[List[ReferencePhraseInfo]],  # Output for: reference_phrases_info_state
    float  # Output for: reference_total_duration_secs_state
]:
    """
    Callback function triggered when a MIDI file is uploaded.
    It processes the uploaded MIDI file by:
    1. Calling `backend_logic.extract_melody_data_from_midi` to get musical features.
    2. If successful, it generates an initial population of melodies based on the reference
       by calling `backend_logic.evolve_one_generation` (simulating a "generation 0").
    3. Updates various Gradio UI components (like tempo, key display, status messages,
       and the dropdown for selecting melodies) and application state variables.

    Args:
        midi_file_obj: The Gradio File object representing the uploaded file.
        current_tempo_gr_val: Tempo value from the UI (used as a fallback if detection fails).
        current_key_gr_val: Key signature from the UI (used as a fallback).

    Returns:
        A tuple containing values to update the corresponding Gradio output components and states.
        The order and types must match the `outputs` list in the `load_midi_button.upload(...)` call.
    """
    # Define a default return tuple for error scenarios to ensure consistent output structure.
    default_return_on_error = (
        [], [], [], current_tempo_gr_val, current_key_gr_val, 0,  # Empty states, retain current UI tempo/key
        "Error during MIDI load. Please check the file or console for details.",  # Status message
        gr.Dropdown(choices=[], value=None),  # Empty dropdown
        None, "0", None, 0.0  # Clear audio, reset gen count, no phrase/duration info
    )

    if midi_file_obj is None:  # No file was uploaded.
        return ([], [], [], current_tempo_gr_val, current_key_gr_val, 0,
                "No MIDI file provided. Please upload a MIDI file to begin.",
                gr.Dropdown(choices=[], value=None), None, "0", None, 0.0)

    # The Gradio File object's `name` attribute holds the temporary path to the uploaded file.
    uploaded_file_path = midi_file_obj.name

    # Call backend logic to extract musical data from the MIDI file.
    # This now also returns the total duration of the reference melody in seconds.
    (extracted_pitches, extracted_rhythms, detected_tempo, detected_key,
     original_filename, error_message_from_backend, ref_phrases_analysis,
     reference_total_duration_seconds  # Newly returned value
     ) = backend_logic.extract_melody_data_from_midi(uploaded_file_path)

    if error_message_from_backend:  # If the backend reported an error during MIDI processing.
        status_update = f"Error loading '{original_filename}': {error_message_from_backend}"
        # Return extracted data if any, otherwise defaults.
        return (extracted_pitches or [], extracted_rhythms or [], [],
                detected_tempo or current_tempo_gr_val,  # Use detected if available, else UI's current
                detected_key or current_key_gr_val,
                0, status_update, gr.Dropdown(choices=[], value=None), None, "0",
                ref_phrases_analysis, reference_total_duration_seconds or 0.0)

    if not extracted_pitches:  # MIDI loaded, but no notes/rests were found.
        status_update = f"Loaded '{original_filename}', but no notes or rests were extracted. Cannot proceed."
        return ([], [], [], detected_tempo, detected_key, 0, status_update,
                gr.Dropdown(choices=[], value=None), None, "0", None, 0.0)

    # --- Generate Initial Population (Generation 0) ---
    # Based on the successfully extracted reference melody, create an initial set of melodies.
    # This is done by calling `evolve_one_generation` with an empty current population,
    # effectively making it produce the first generation (population).
    initial_population: List[MelodySequence] = backend_logic.evolve_one_generation(
        current_population=[],  # Start with no prior population for initial generation
        ref_pitches=extracted_pitches,
        ref_rhythms=extracted_rhythms,
        current_key_name=detected_key,
        ref_phrases_info=ref_phrases_analysis,
        tempo_bpm=detected_tempo,  # Pass detected tempo for GA context
        ref_total_duration_secs=reference_total_duration_seconds  # Pass total duration for GA context
    )

    if not initial_population and extracted_pitches:  # Should be rare if evolve_one_generation has fallbacks
        status_update = f"Loaded '{original_filename}', but failed to generate an initial population. Check console."
        return (extracted_pitches, extracted_rhythms, [], detected_tempo, detected_key, 0, status_update,
                gr.Dropdown(choices=[], value=None), None, "0", ref_phrases_analysis, reference_total_duration_seconds)

    # Prepare choices for the melody selection dropdown menu.
    dropdown_choices = [f"Initial Melody {i + 1}" for i in range(len(initial_population))]
    status_update_message = (
        f"Successfully loaded: {original_filename}\n"
        f"Key: {detected_key}, Tempo: {detected_tempo} BPM, Events: {len(extracted_pitches)}, Duration: {reference_total_duration_seconds:.2f}s.\n"
        f"Initial population of {len(initial_population)} melodies generated (Generation 0).\n"
        f"Reference phrases analyzed: {len(ref_phrases_analysis) if ref_phrases_analysis else 'N/A'}"
    )
    current_generation_count = 0  # Reset generation counter on new MIDI load.

    # Return all values to update the UI and application state.
    return (
        extracted_pitches, extracted_rhythms, initial_population,
        detected_tempo, detected_key, current_generation_count,
        status_update_message,
        gr.Dropdown(choices=dropdown_choices, value=dropdown_choices[0] if dropdown_choices else None),
        # Select first melody by default
        None,  # Clear audio player
        str(current_generation_count),  # Update generation display
        ref_phrases_analysis,  # Store the analyzed phrase info from reference
        reference_total_duration_seconds  # Store the total duration in seconds
    )


def handle_evolve_melodies(
        ref_pitches_state: List[Optional[str]],  # Stored pitch sequence from the reference MIDI.
        ref_rhythms_state: List[float],  # Stored rhythm sequence from the reference MIDI.
        current_pop_state: List[MelodySequence],  # The current population of evolved melodies.
        gen_count_state: int,  # The current generation number.
        key_sig_from_display: str,  # Key signature currently shown in UI.
        tempo_from_input: int,  # Tempo from UI (this is the tempo_bpm for GA).
        ref_phrases_info_state: Optional[List[ReferencePhraseInfo]],  # Stored phrase info.
        ref_total_duration_secs_state: float  # Stored total duration of reference in seconds.
) -> Tuple[  # Defines the structure of the tuple returned to update Gradio outputs.
    List[MelodySequence],  # Output for: current_evolved_melodies_state (new population)
    int,  # Output for: generation_count_state (incremented)
    str,  # Output for: status_display
    gr.Dropdown,  # Output for: evolved_melody_selector
    Optional[str],  # Output for: audio_player (cleared)
    str  # Output for: generation_display
]:
    """
    Callback for the "Evolve New Melodies" button.
    Triggers one generation of the genetic algorithm using `backend_logic.evolve_one_generation`,
    passing all necessary reference data, including tempo and total duration.

    Args:
        (various state and UI input values as listed above)

    Returns:
        A tuple of values to update corresponding Gradio components and state variables.
    """
    current_key_for_evolution = key_sig_from_display  # Use key from UI display.

    # Validate that reference data is available before attempting evolution.
    if not ref_pitches_state or not ref_rhythms_state:
        status_update = "Cannot evolve: No reference MIDI loaded or essential data is missing."
        # Keep current state for dropdown if it exists from a previous valid load.
        old_dropdown_choices = [f"Gen {gen_count_state} - Melody {i + 1}" for i in
                                range(len(current_pop_state))] if current_pop_state else []
        return (current_pop_state, gen_count_state, status_update,
                gr.Dropdown(choices=old_dropdown_choices,
                            value=old_dropdown_choices[0] if old_dropdown_choices else None),
                None, str(gen_count_state))

    # Call backend logic to run one generation of evolution.
    # Crucially, pass `tempo_from_input` as `tempo_bpm` and `ref_total_duration_secs_state`.
    new_evolved_population = backend_logic.evolve_one_generation(
        current_pop_state, ref_pitches_state, ref_rhythms_state,
        current_key_for_evolution, ref_phrases_info_state,
        tempo_bpm=tempo_from_input,  # Pass the tempo from UI input.
        ref_total_duration_secs=ref_total_duration_secs_state  # Pass stored total duration.
    )

    if not new_evolved_population:  # Handle case where evolution fails to produce a valid new population.
        old_dropdown_choices = [f"Gen {gen_count_state} - Melody {i + 1}" for i in
                                range(len(current_pop_state))] if current_pop_state else []
        status_update = "Evolution failed to produce a new valid population. Current population retained. Check console for errors."
        return (current_pop_state, gen_count_state, status_update,
                gr.Dropdown(choices=old_dropdown_choices,
                            value=old_dropdown_choices[0] if old_dropdown_choices else None),
                None, str(gen_count_state))

    updated_generation_count = gen_count_state + 1
    # Prepare choices for the melody selection dropdown, reflecting the new generation.
    new_dropdown_choices = [f"Gen {updated_generation_count} - Melody {i + 1}" for i in
                            range(len(new_evolved_population))]
    status_update_message = f"Evolved to Generation {updated_generation_count}. Population size: {len(new_evolved_population)}."

    return (
        new_evolved_population, updated_generation_count, status_update_message,
        gr.Dropdown(choices=new_dropdown_choices, value=new_dropdown_choices[0] if new_dropdown_choices else None),
        # Select first melody
        None,  # Clear audio player on new evolution, user must re-select.
        str(updated_generation_count)  # Update generation display.
    )


def handle_play_selected_melody(
        selected_melody_choice_str: Optional[str],  # String from the dropdown (e.g., "Gen 1 - Melody 5")
        evolved_melodies_list_state: List[MelodySequence],  # Current list of all evolved melodies
        tempo_val_from_input: int,  # Current tempo from UI for audio synthesis
        key_sig_from_display: str  # Current key signature from UI for audio synthesis
) -> Tuple[Optional[str], str]:  # Returns (path_to_mp3_audio_file_or_None, status_message_string)
    """
    Callback triggered when a melody is selected from the dropdown menu.
    It generates an MP3 audio file for the selected melody using `backend_logic`
    and updates the Gradio audio player component to play it.

    Args:
        selected_melody_choice_str: The string identifying the chosen melody.
        evolved_melodies_list_state: The current list of all evolved melodies.
        tempo_val_from_input: Current tempo from UI.
        key_sig_from_display: Current key signature from UI.

    Returns:
        A tuple: (path_to_mp3_audio_file, status_message).
                 The path is None if MP3 generation fails.
    """
    generated_mp3_path: Optional[str] = None  # Path for the audio player.
    status_message = "Processing selection for playback..."

    if not selected_melody_choice_str or not evolved_melodies_list_state:
        status_message = "No melody selected or no melodies are available to play."
        return generated_mp3_path, status_message

    try:
        # Extract the melody index from the dropdown string (e.g., "Gen X - Melody Y" -> index Y-1).
        # This parsing needs to be robust to the label format used in dropdown choices.
        parts = selected_melody_choice_str.split("Melody ")
        if len(parts) < 2 or not parts[-1].strip().isdigit():  # Check if last part is a number.
            raise ValueError("Invalid melody selection string format for index extraction.")
        melody_index = int(parts[-1].strip()) - 1  # Convert to 0-based index.
    except ValueError as e:
        status_message = f"Error identifying selected melody from '{selected_melody_choice_str}': {e}"
        return generated_mp3_path, status_message

    # Validate the extracted index.
    if not (0 <= melody_index < len(evolved_melodies_list_state)):
        status_message = (f"Selected melody index {melody_index + 1} is out of bounds for the current "
                          f"population of {len(evolved_melodies_list_state)} melodies.")
        return generated_mp3_path, status_message

    selected_melody_data: MelodySequence = evolved_melodies_list_state[melody_index]

    try:
        current_tempo_for_playback = int(tempo_val_from_input)  # Ensure tempo is integer.
    except ValueError:
        status_message = "Invalid tempo value provided. Using default 120 BPM for playback."
        current_tempo_for_playback = 120  # Fallback tempo.

    # Call backend logic to convert the selected melody to an MP3 file.
    generated_mp3_path = backend_logic.convert_melody_to_mp3_file(
        selected_melody_data, current_tempo_for_playback, key_sig_from_display,
        base_filename="playback_audio"  # Temporary filename base for the MP3.
    )

    if generated_mp3_path and os.path.exists(generated_mp3_path):
        status_message = f"Audio ready for playback: {selected_melody_choice_str}"
    else:
        status_message = f"Failed to generate audio for {selected_melody_choice_str}. Please check the console for errors."
        generated_mp3_path = None  # Ensure path is None if MP3 generation failed.

    return generated_mp3_path, status_message


def handle_save_selected_file(
        file_type_to_save: str,  # String indicating "MIDI" or "MP3".
        selected_melody_choice_str: Optional[str],  # String from the melody selection dropdown.
        evolved_melodies_list_state: List[MelodySequence],  # Current list of evolved melodies.
        tempo_val_from_input: int,  # Current tempo from UI.
        key_sig_from_display: str  # Current key signature from UI.
) -> Tuple[Optional[str], str]:  # Returns (path_to_generated_file_for_download_or_None, status_message_string)
    """
    Generic handler for saving the selected evolved melody as either a MIDI or MP3 file.
    This function is called by the specific MIDI and MP3 save button handlers, differentiated
    by the `file_type_to_save` argument.

    Args:
        file_type_to_save: String "MIDI" or "MP3" indicating the desired output format.
        (other arguments are similar to handle_play_selected_melody)

    Returns:
        A tuple: (path_to_generated_file, status_message).
                 The path is used by Gradio's `gr.File` component to trigger a download.
                 Path is None if file generation fails.
    """
    output_file_path_for_download: Optional[str] = None  # Path for the gr.File component.
    status_message = f"Processing request to save selected melody as {file_type_to_save}..."

    # Validate inputs.
    if not selected_melody_choice_str or not evolved_melodies_list_state:
        status_message = f"No melody selected. Cannot save as {file_type_to_save}."
        return output_file_path_for_download, status_message

    try:  # Parse melody index from dropdown string.
        parts = selected_melody_choice_str.split("Melody ")
        if len(parts) < 2 or not parts[-1].strip().isdigit():
            raise ValueError("Invalid melody selection string format for index extraction.")
        melody_index = int(parts[-1].strip()) - 1
    except ValueError as e:
        status_message = f"Error identifying selected melody for {file_type_to_save} download: {e}"
        return output_file_path_for_download, status_message

    if not (0 <= melody_index < len(evolved_melodies_list_state)):  # Validate index.
        status_message = f"Selected melody index {melody_index + 1} is out of bounds for {file_type_to_save} download."
        return output_file_path_for_download, status_message

    selected_melody_data: MelodySequence = evolved_melodies_list_state[melody_index]

    try:  # Validate and get tempo.
        current_tempo_for_file = int(tempo_val_from_input)
    except ValueError:
        status_message = f"Invalid tempo value. Using default 120 BPM for {file_type_to_save} file."
        current_tempo_for_file = 120  # Fallback tempo.

    # --- Generate a descriptive base filename for the downloaded file ---
    # Sanitize parts of the filename for broader OS compatibility.
    safe_key_signature_part = key_sig_from_display.replace(" ", "_").replace("#", "sharp").replace("b", "flat").replace(
        "/", "_")

    # Extract generation info and melody number carefully.
    generation_info_parts = selected_melody_choice_str.split(" - ")
    generation_identifier_part = generation_info_parts[0].replace(" ",
                                                                  "") if generation_info_parts and " - " in selected_melody_choice_str else "Initial"

    melody_number_parts = selected_melody_choice_str.split("Melody ")
    melody_number_identifier_part = melody_number_parts[-1].strip() if melody_number_parts and len(
        melody_number_parts) > 1 else "Unknown"

    suggested_download_filename_base = f"{generation_identifier_part}_Melody{melody_number_identifier_part}_{safe_key_signature_part}_{current_tempo_for_file}bpm"

    # --- Call appropriate backend function based on `file_type_to_save` ---
    if file_type_to_save == "MIDI":
        output_file_path_for_download = backend_logic.save_melody_to_midi_file(
            selected_melody_data, current_tempo_for_file, key_sig_from_display,
            base_filename=suggested_download_filename_base  # Pass the generated base filename.
        )
    elif file_type_to_save == "MP3":
        output_file_path_for_download = backend_logic.convert_melody_to_mp3_file(
            selected_melody_data, current_tempo_for_file, key_sig_from_display,
            base_filename=suggested_download_filename_base + "_audio"  # Add suffix for MP3.
        )
    else:  # Should not happen if called correctly from UI.
        status_message = f"Unsupported file type specified for saving: {file_type_to_save}"
        return None, status_message

    # Update status message based on success or failure of file generation.
    if output_file_path_for_download and os.path.exists(output_file_path_for_download):
        status_message = f"{file_type_to_save} file for '{selected_melody_choice_str}' has been prepared for download."
    else:
        status_message = f"Failed to generate {file_type_to_save} file for '{selected_melody_choice_str}'. Please check console for errors."
        output_file_path_for_download = None  # Ensure path is None if generation failed.

    return output_file_path_for_download, status_message


# --- Gradio UI Definition ---
# `gr.Blocks` allows for more complex and custom layouts compared to `gr.Interface`.
with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue)) as demo:
    gr.Markdown("# 🎵 MIDI Melody Evolver Pro 🎶")
    gr.Markdown(
        "Upload a reference MIDI file. The application will then use a genetic algorithm "
        "to evolve new melodies that aim to resemble the reference in structure, duration, and key phrase points, "
        "while also incorporating general musical coherence. You can play back and download the results."
    )

    # --- Application State Variables (Hidden UI Components) ---
    # These `gr.State` objects hold data that persists across user interactions within a session.
    reference_pitches_state = gr.State([])  # Stores pitches from the reference MIDI.
    reference_rhythms_state = gr.State([])  # Stores rhythms (durations) from the reference MIDI.
    current_evolved_melodies_state = gr.State([])  # Stores the current population of evolved melodies.
    generation_count_state = gr.State(0)  # Tracks the number of generations evolved.
    reference_phrases_info_state = gr.State(None)  # Stores analyzed ReferencePhraseInfo data from the reference MIDI.
    reference_total_duration_secs_state = gr.State(0.0)  # Stores total duration of reference MIDI in seconds.

    # --- UI Layout Definition ---
    with gr.Row():  # Main row for input controls and status displays.
        with gr.Column(scale=1):  # Left column for MIDI loading and parameters.
            gr.Markdown("### 1. Load Reference MIDI")
            load_midi_button = gr.UploadButton(
                "📁 Load Reference MIDI File",
                file_types=[".mid", ".midi"],  # Specify allowed file types for upload.
            )
            tempo_input = gr.Number(label="Tempo (BPM)", value=120, interactive=True, step=1, precision=0,
                                    info="Detected from MIDI, or set manually for playback/export.")
            key_display = gr.Textbox(label="Detected Key Signature", value="N/A", interactive=False,
                                     info="Key signature analyzed from the uploaded MIDI.")

        with gr.Column(scale=2):  # Right column for status messages and generation info.
            gr.Markdown("### Application Status & Log")
            status_display = gr.Textbox(
                label="Log & Status Messages",
                value="Please load a MIDI file to begin the evolutionary process.",
                interactive=False,  # User cannot type here.
                lines=5,  # Initial number of visible lines.
                max_lines=10  # Max lines before scrolling.
            )
            generation_display = gr.Textbox(label="Current Generation Evolved", value="0", interactive=False)

    gr.Markdown("---")  # Visual separator.

    gr.Markdown("### 2. Evolve Melodies")
    evolve_button = gr.Button("🧬 Evolve New Melodies (Run Next Generation)", variant="primary")

    gr.Markdown("---")  # Visual separator.

    gr.Markdown("### 3. Manage and Export Evolved Melodies")
    evolved_melody_selector = gr.Dropdown(
        label="Select Evolved Melody for Playback / Download",
        choices=[],  # Initially empty; populated after MIDI load and evolution.
        interactive=True,
        info="Choose a melody from the current evolved population."
    )
    # Removed 'info' argument from gr.Audio as it caused the TypeError
    audio_player = gr.Audio(label="Melody Playback (MP3)", type="filepath", interactive=False)

    with gr.Row():  # Row for download buttons.
        save_midi_button = gr.Button("💾 Save Selected Melody as MIDI")
        save_mp3_button = gr.Button("💾 Save Selected Melody as MP3")

    # Hidden `gr.File` components are used to trigger file downloads in Gradio.
    download_midi_output_file = gr.File(label="Download MIDI File", interactive=False)
    download_mp3_output_file = gr.File(label="Download MP3 File", interactive=False)

    # --- Event Handling: Connecting UI components to their respective callback functions ---

    load_midi_button.upload(
        fn=handle_load_midi,
        inputs=[load_midi_button, tempo_input, key_display],
        outputs=[
            reference_pitches_state, reference_rhythms_state,
            current_evolved_melodies_state, tempo_input, key_display,
            generation_count_state, status_display, evolved_melody_selector,
            audio_player, generation_display, reference_phrases_info_state,
            reference_total_duration_secs_state
        ]
    )

    evolve_button.click(
        fn=handle_evolve_melodies,
        inputs=[
            reference_pitches_state, reference_rhythms_state,
            current_evolved_melodies_state, generation_count_state,
            key_display, tempo_input,  # Pass current key and tempo from UI.
            reference_phrases_info_state, reference_total_duration_secs_state
        ],
        outputs=[  # Update state and UI after evolution.
            current_evolved_melodies_state, generation_count_state, status_display,
            evolved_melody_selector, audio_player, generation_display
        ]
    )

    # Event: Selection in `evolved_melody_selector` (dropdown) changes.
    evolved_melody_selector.change(
        fn=handle_play_selected_melody,
        inputs=[evolved_melody_selector, current_evolved_melodies_state, tempo_input, key_display],
        outputs=[audio_player, status_display]  # Update audio player and status.
    )

    # Event: `save_midi_button` is clicked.
    # A lambda function is used to pass the additional `file_type_to_save` argument.
    save_midi_button.click(
        fn=lambda sel_choice, pop_state, tempo_ui, key_ui: handle_save_selected_file(
            "MIDI", sel_choice, pop_state, tempo_ui, key_ui
        ),
        inputs=[evolved_melody_selector, current_evolved_melodies_state, tempo_input, key_display],
        outputs=[download_midi_output_file, status_display]  # Output to hidden file component and status.
    )

    # Event: `save_mp3_button` is clicked.
    save_mp3_button.click(
        fn=lambda sel_choice, pop_state, tempo_ui, key_ui: handle_save_selected_file(
            "MP3", sel_choice, pop_state, tempo_ui, key_ui
        ),
        inputs=[evolved_melody_selector, current_evolved_melodies_state, tempo_input, key_display],
        outputs=[download_mp3_output_file, status_display]  # Output to hidden file component and status.
    )

# --- Launch the Gradio Application ---
if __name__ == "__main__":
    print("Launching Gradio MIDI Melody Evolver...")
    # `debug=True` enables more detailed error messages in the console during development.
    # `inbrowser=True` attempts to open the app in a new browser tab automatically upon launch.
    # `share=True` (if uncommented) would generate a public Gradio link (requires internet).
    demo.launch(debug=True, inbrowser=True)
