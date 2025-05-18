# ga_components/melody_generator.py
"""
Module: melody_generator.py

Purpose:
This module is responsible for the procedural generation and modification
(mutation) of musical melodies within the genetic algorithm. It uses musical
rules, contextual information (like key and previous notes), and probabilistic
choices to create initial melodies and to introduce variations during the
evolutionary process. The goal is to produce musically plausible sequences
that can then be evaluated and evolved by the GeneticAlgorithmCore.

Key Functionalities:
- Initialization: Sets up the generator with a specific musical key, allowed pitches,
  and tonal context.
- Melody Generation (`generate_melody`): Creates new melodies from scratch. This process
  is guided by a target number of musical events (notes/rests) and can also take
  reference phrase durations to structure the generated melody similarly to a reference.
  It builds melodies phrase by phrase, making decisions about pitch and rhythm for
  each note based on musical heuristics (e.g., favoring stepwise motion, using
  tonally important notes, avoiding excessive repetition).
- Melody Mutation (`mutate_melody`): Modifies an existing melody by introducing small,
  random changes. This is a key part of the GA's exploration of the solution space.
  Mutations can include:
    - Changing a note's pitch.
    - Altering a note's duration.
    - Changing a note to a rest, or a rest to a note.
  Mutations are also performed probabilistically and with consideration for musical context.
- Motif Development (`_apply_motif_development`): A helper function to apply simple
  transformations (like transposition or rhythmic variation) to a segment of a melody,
  used for creating internal repetition with variation during melody generation.

Design Philosophy:
The MelodyGenerator aims to be a "musically aware" random source. It doesn't directly
evaluate melodies against a full reference (that's the GA's fitness function's job).
Instead, it uses localized rules and probabilities to ensure that the melodies it
produces or mutates are generally well-formed and have a higher chance of being
musically interesting or useful as building blocks for the GA.
"""

import random
from typing import List, Tuple, Optional, Dict, Set  # Dict, Set are used for internal structures

# Third-party library imports (music21 for music theory objects)
from music21 import note, interval, pitch as m21pitch, key  # Added key for type hint consistency

# Local application/library specific imports
# Import constants defining musical parameters, generation probabilities, etc.
from .music_constants import (
    MIN_OCTAVE, MAX_OCTAVE, POSSIBLE_DURATIONS,
    MOTIF_REPETITION_CHANCE, INTERNAL_PHRASE_REPETITION_CHANCE,
    PHRASE_PUNCTUATION_CHANCE, REST_PROBABILITY_GENERATION,
    MAX_CONSECUTIVE_REPEATED_PITCHES, MelodySequence, MelodyNote,
    NOTES_PER_PHRASE_FALLBACK, DEFAULT_PHRASE_DURATION_BEATS,
    MUTATION_RATE,  # Default mutation rate for the mutate_melody method
    REST_PROBABILITY_MUTATION, PHRASE_END_MUTATION_CHANCE, MIN_NOTES_PER_PHRASE, ACCEPTABLE_LEAP_INTERVAL
)
# Import utility functions for music analysis (e.g., key parsing, scale generation).
from .music_utils import MusicUtils


class MelodyGenerator:
    """
    Generates and mutates melodies based on musical rules, a given key,
    and probabilistic choices. It aims to produce musically plausible sequences
    that can be evolved by the genetic algorithm.
    """

    def __init__(self, key_name_str: str,
                 notes_per_phrase_fallback: int = NOTES_PER_PHRASE_FALLBACK,
                 target_phrase_duration_beats: float = DEFAULT_PHRASE_DURATION_BEATS):
        """
        Initializes the MelodyGenerator with a musical key and default phrase parameters.

        Args:
            key_name_str (str): The musical key for generation (e.g., "C major", "g minor").
                                This string is parsed into a music21.key.Key object.
            notes_per_phrase_fallback (int): Default number of notes/events per phrase if generation
                                             is not guided by specific reference phrase durations.
            target_phrase_duration_beats (float): Default target duration for generated phrases
                                                  in beats, if not guided by reference.
        """
        self.key_name: str = key_name_str
        # Parse the key string into a music21 Key object for music theory operations.
        self.key_obj: key.Key = MusicUtils.get_key_object(key_name_str)

        # Determine the set of allowed pitches (pitch name with octave) within the
        # specified key and the defined octave range (from music_constants).
        self.allowed_pitches: List[str] = MusicUtils.get_scale_pitches_for_key(
            key_name_str, (MIN_OCTAVE, MAX_OCTAVE)
        )
        # Get important tonal centers (tonic, dominant, subdominant, etc.) for the current key.
        # These are used to weight pitch choices during generation.
        self.tonal_nodes: Dict[str, Optional[str]] = MusicUtils.get_tonal_hierarchy(self.key_obj)

        # Pre-calculate sets of chord tones for common diatonic chords in the key.
        # This is used by `_is_justified_leap` to make melodic leaps sound more natural
        # if they outline a common chord.
        self.common_chord_tones_sets: List[Set[str]] = [
            MusicUtils.get_chord_tones(self.key_obj, degree) for degree in [1, 2, 3, 4, 5, 6]  # Common diatonic degrees
            if MusicUtils.get_chord_tones(self.key_obj, degree)  # Ensure the chord tone set is not empty
        ]

        self.notes_per_phrase_fallback: int = notes_per_phrase_fallback
        self.target_phrase_duration_beats: float = target_phrase_duration_beats

        # Create a weighted list of allowed pitches. Tonally important notes (tonic, dominant)
        # appear more frequently in this list, increasing their probability of being chosen
        # during random pitch selection.
        self.weighted_pitches: List[str] = self._create_weighted_pitches()

        # Fallbacks: If, for some reason, allowed_pitches or weighted_pitches are empty
        # (e.g., due to an extremely unusual key or error in MusicUtils), provide defaults
        # to prevent crashes. This should be rare given MusicUtils' own fallbacks.
        if not self.allowed_pitches:
            print(
                f"Warning (MelodyGenerator): No allowed pitches found for key '{key_name_str}'. Defaulting to C major pitches within range.")
            self.allowed_pitches = [p_name + str(octv) for octv in range(MIN_OCTAVE, MAX_OCTAVE + 1) for p_name in
                                    "CDEFGAB"]
        if not self.weighted_pitches:
            self.weighted_pitches = list(self.allowed_pitches)  # Use unweighted list if weighting failed

    def _create_weighted_pitches(self) -> List[str]:
        """
        Internal helper method to create a list of allowed pitches where tonally important
        notes (tonic, dominant, subdominant, mediant) are repeated according to predefined
        weights. This biases random pitch selection towards these more stable scale degrees.

        Returns:
            List[str]: A weighted list of pitch strings (e.g., "C4" might appear 8 times,
                       while "B4" might appear only once).
        """
        weighted_list: List[str] = []
        if not self.allowed_pitches:  # Should be caught by constructor fallback
            return ['C4']  # Minimal fallback if allowed_pitches is somehow still empty

        for pitch_name_with_octave in self.allowed_pitches:
            try:
                # We need the pitch class (e.g., "C", "F#") to compare with tonal_nodes.
                current_pitch_obj = m21pitch.Pitch(pitch_name_with_octave)
                pitch_class_name = current_pitch_obj.name  # e.g., "C" from "C4"

                weight = 1  # Default weight for less emphasized scale degrees

                # Assign higher weights to more stable/important scale degrees.
                # These weights are heuristics and can be tuned.
                if self.tonal_nodes.get("tonic") and pitch_class_name == self.tonal_nodes["tonic"]:
                    weight = 8
                elif self.tonal_nodes.get("dominant") and pitch_class_name == self.tonal_nodes["dominant"]:
                    weight = 6
                elif self.tonal_nodes.get("subdominant") and pitch_class_name == self.tonal_nodes["subdominant"]:
                    weight = 4
                elif self.tonal_nodes.get("mediant") and pitch_class_name == self.tonal_nodes["mediant"]:
                    weight = 3
                # Other degrees (supertonic, submediant, leading_tone) get the default weight of 1.

                # Extend the list by adding the current pitch 'weight' number of times.
                weighted_list.extend([pitch_name_with_octave] * weight)
            except Exception as e:
                # If parsing the pitch string fails (unlikely for validly generated allowed_pitches),
                # add it with default weight to avoid losing the pitch.
                print(
                    f"Warning (MelodyGenerator._create_weighted_pitches): Could not parse pitch '{pitch_name_with_octave}' for weighting: {e}")
                weighted_list.append(pitch_name_with_octave)

        # If the weighting process somehow resulted in an empty list, fall back to the unweighted allowed_pitches.
        return weighted_list if weighted_list else list(self.allowed_pitches)

    def _is_justified_leap(self, prev_pitch_obj: Optional[m21pitch.Pitch],
                           current_pitch_obj: Optional[m21pitch.Pitch]) -> bool:
        """
        Internal helper to determine if a melodic leap (interval larger than a step)
        between two pitches can be considered "musically justified."
        A common justification is if both notes belong to the same underlying diatonic chord
        in the current key (e.g., leaping between C and G in C major, as both are in the C major triad).

        Args:
            prev_pitch_obj: The preceding music21.pitch.Pitch object (can be None).
            current_pitch_obj: The current music21.pitch.Pitch object (can be None).

        Returns:
            bool: True if the leap is considered justified, False otherwise.
        """
        if not prev_pitch_obj or not current_pitch_obj:
            return False  # Cannot justify a leap if one of the pitches is missing or a rest.

        # Check if the pitch classes (names) of both pitches are present in any of the
        # pre-calculated common chord tone sets for the current key.
        for chord_tone_set in self.common_chord_tones_sets:
            if prev_pitch_obj.name in chord_tone_set and current_pitch_obj.name in chord_tone_set:
                return True  # Leap is justified as both notes belong to a common chord.
        return False  # Leap is not clearly justified by this heuristic.

    def _get_next_note_tuple(self,
                             previous_pitch_obj: Optional[m21pitch.Pitch],
                             last_leap_semitones: int,
                             time_left_in_phrase: float,
                             is_phrase_end_approaching: bool,
                             current_phrase_duration_beats: float,
                             target_duration_this_phrase: float
                             ) -> MelodyNote:
        """
        Core logic for selecting the next single musical event (pitch and duration)
        when generating a melody. It considers musical context such as the previous note,
        melodic contour, key signature, and remaining time in the current phrase.
        It also includes a probability of generating a rest instead of a note.

        Args:
            previous_pitch_obj: The music21.pitch.Pitch of the preceding note.
                                None if at the beginning of a melody or after a rest.
            last_leap_semitones: The size (in semitones) of the last melodic leap taken.
                                 Positive for upward, negative for downward. Used for contour correction.
            time_left_in_phrase: Remaining duration in beats for the current phrase being generated.
            is_phrase_end_approaching: Boolean indicating if the phrase is near its target duration.
                                       Influences choice of rests and note durations.
            current_phrase_duration_beats: Accumulated duration of notes/rests already in the current phrase.
            target_duration_this_phrase: Target total duration for the current phrase.

        Returns:
            MelodyNote: A tuple (pitch_string_or_None, duration_float).
                        Pitch is None if a rest is generated.
        """
        chosen_pitch_str: Optional[str]  # Will hold the selected pitch string or None for a rest.

        # --- Pitch Selection Logic ---
        # Decide probabilistically whether to generate a note or a rest.
        # Rests are less likely if the phrase is about to end, to avoid abrupt cutoffs.
        if random.random() < REST_PROBABILITY_GENERATION and not is_phrase_end_approaching:
            chosen_pitch_str = None  # Generate a rest.
        else:  # Generate a note.
            if previous_pitch_obj is None or not self.allowed_pitches:
                # If there's no previous pitch (start of melody/phrase or after a rest),
                # or if allowed_pitches list is somehow empty, choose randomly from the weighted list.
                chosen_pitch_str = random.choice(
                    self.weighted_pitches if self.weighted_pitches else ['C4']  # Fallback to C4
                )
            else:
                # If there is a previous pitch, apply melodic contour logic to select the next one.
                # This involves weighting potential next pitches based on interval size,
                # justification of leaps, and correction for previous large leaps.
                candidate_pitches_weighted: List[Tuple[str, int]] = []
                for p_str_candidate in self.allowed_pitches:
                    try:
                        candidate_obj = m21pitch.Pitch(p_str_candidate)
                        # Basic filter: avoid excessively large octave jumps unless musically intended.
                        # Check if octave difference is more than 1 AND semitone difference is more than an octave.
                        if abs(candidate_obj.octave - previous_pitch_obj.octave) > 1 and \
                                abs(candidate_obj.ps - previous_pitch_obj.ps) > 12:
                            continue  # Skip this candidate as it's too far.

                        interval_to_candidate = interval.Interval(previous_pitch_obj, candidate_obj)
                        interval_semitones = interval_to_candidate.semitones
                        abs_interval = abs(interval_semitones)

                        weight = 0  # Base weight for this candidate.

                        # Weighting based on interval size (favoring smaller intervals like steps).
                        if abs_interval == 0:  # Repeated pitch (same pitch class and octave)
                            weight = 1  # Low weight; explicit repetition handled by MAX_CONSECUTIVE_REPEATED_PITCHES logic.
                        elif abs_interval <= 2:  # Stepwise motion (minor/major second)
                            weight = 30  # Strongly favored.
                        elif abs_interval <= 4:  # Small leap (minor/major third)
                            weight = 10
                        elif abs_interval <= ACCEPTABLE_LEAP_INTERVAL:  # Medium leap (up to a perfect fifth, or configured interval)
                            if self._is_justified_leap(previous_pitch_obj, candidate_obj):
                                weight = 5  # Justified leaps are more acceptable.
                            else:
                                weight = 2  # Unjustified medium leaps less favored.
                        else:  # Large leap (greater than ACCEPTABLE_LEAP_INTERVAL)
                            weight = 1  # Generally disfavored unless no other options.

                        # Contour correction: If the last melodic move was a large leap,
                        # favor stepwise motion in the opposite direction to balance the contour.
                        if abs(last_leap_semitones) > ACCEPTABLE_LEAP_INTERVAL:  # If last leap was large
                            # Check if current candidate is stepwise AND in opposite direction of last leap
                            if abs_interval <= 2 and (interval_semitones * last_leap_semitones < 0):
                                weight *= 3  # Strongly favor this corrective motion.

                        if weight > 0:
                            candidate_pitches_weighted.append((p_str_candidate, int(weight)))
                    except Exception:
                        continue  # Skip problematic pitch candidates (e.g., parsing error).

                if not candidate_pitches_weighted:
                    # Fallback if no candidates meet criteria (e.g., very restrictive previous pitch).
                    # Choose randomly from the general weighted list.
                    chosen_pitch_str = random.choice(
                        self.weighted_pitches if self.weighted_pitches else ['C4']
                    )
                else:
                    # Choose from the generated candidates based on their calculated weights.
                    choices, weights_values = zip(*candidate_pitches_weighted)
                    chosen_pitch_str = random.choices(choices, weights=weights_values, k=1)[0]

        # --- Duration Selection Logic ---
        # Filter possible durations to those that fit within the remaining time for the phrase.
        # Allow a slight overshoot (+0.25 beats) which can be implicitly trimmed if it's the last note.
        possible_rhythms_for_time_left = [r for r in POSSIBLE_DURATIONS if r <= time_left_in_phrase + 0.25]
        chosen_duration: float

        if not possible_rhythms_for_time_left:
            # If no standard rhythm fits (e.g., very small time_left), use the shortest possible duration
            # or fill the remaining time if it's extremely short.
            chosen_duration = min(POSSIBLE_DURATIONS) if time_left_in_phrase > 0.1 else max(0.125,
                                                                                            time_left_in_phrase)  # Ensure some duration
        elif is_phrase_end_approaching:
            # If nearing the end of the phrase, try to pick a duration that fits the remaining time well,
            # or choose a common ending rhythm (often longer, like a half or whole note, if time allows).
            fitting_rhythms = [r for r in possible_rhythms_for_time_left if
                               abs(r - time_left_in_phrase) < 0.125]  # Rhythms that nearly fill the gap
            if fitting_rhythms and random.random() < 0.7:  # High chance to fit exactly
                chosen_duration = random.choice(fitting_rhythms)
            else:  # Otherwise, choose from available options, perhaps slightly longer to fill.
                chosen_duration = random.choice(possible_rhythms_for_time_left)
        elif current_phrase_duration_beats < target_duration_this_phrase * 0.4:  # Early in the phrase
            # Favor shorter to medium durations to build momentum.
            chosen_duration = random.choice(
                [d for d in possible_rhythms_for_time_left if d <= 1.0] or possible_rhythms_for_time_left)
        else:  # Mid-phrase
            chosen_duration = random.choice(possible_rhythms_for_time_left)

        # Ensure a minimum sensible duration (e.g., a 16th note or 0.25 quarter lengths).
        chosen_duration = max(0.25, chosen_duration)

        # If a rest was chosen (chosen_pitch_str is None), shorter durations are generally more common
        # for rests within a phrase, unless it's a deliberate long pause at a phrase end.
        if chosen_pitch_str is None and not is_phrase_end_approaching:
            chosen_duration = random.choice(
                [d for d in POSSIBLE_DURATIONS if d <= 1.0])  # Favor shorter rests (<= quarter note)

        return (chosen_pitch_str, chosen_duration)

    def _apply_motif_development(self, motif_to_develop: MelodySequence) -> MelodySequence:
        """
        Applies simple transformations (e.g., transposition, rhythmic variation) to a given
        melodic motif to create a "developed" or varied version. This is used internally
        during melody generation to introduce some thematic coherence via repetition with variation.

        Args:
            motif_to_develop (MelodySequence): The original sequence of (pitch, duration) tuples forming the motif.

        Returns:
            MelodySequence: A new MelodySequence representing the developed motif.
                            Returns a copy of the original motif if development is trivial or fails.
        """
        if not motif_to_develop: return []  # Cannot develop an empty motif.

        developed_motif = list(motif_to_develop)  # Work on a copy to avoid modifying the original.
        rand_variation_type = random.random()  # Determine which type of variation to apply.

        if rand_variation_type < 0.6:  # Transposition (60% chance)
            # Transpose the motif by a small, musically common interval.
            # Avoid large transpositions that might go out of comfortable range or key context easily.
            # Intervals in semitones: minor 3rd down/up, major 2nd down/up, perfect 4th down/up (less common for simple variation).
            interval_choices = [-4, -3, -2, 2, 3, 4]
            semitones_to_transpose = random.choice(interval_choices)

            temp_transposed_motif: MelodySequence = []
            is_transposition_valid = True  # Flag to track if all notes transpose successfully.

            for pitch_str, duration in developed_motif:
                if pitch_str is None:  # Keep rests as they are (no pitch to transpose).
                    temp_transposed_motif.append((None, duration))
                    continue
                try:
                    original_pitch_obj = m21pitch.Pitch(pitch_str)
                    transposed_pitch_obj = original_pitch_obj.transpose(semitones_to_transpose)

                    # Check if transposed pitch is within the allowed octave range and
                    # if its pitch class is generally allowed in the current key (simplified check).
                    # A full key context check here would be more complex (e.g., using MusicUtils.get_scale_pitches_for_key).
                    if MIN_OCTAVE <= transposed_pitch_obj.octave <= MAX_OCTAVE and \
                            any(allowed_pitch_name.startswith(transposed_pitch_obj.name) for allowed_pitch_name in
                                self.allowed_pitches):
                        temp_transposed_motif.append((transposed_pitch_obj.nameWithOctave, duration))
                    else:
                        # Transposition resulted in an out-of-range or non-diatonic-sounding note.
                        is_transposition_valid = False
                        break  # Stop this transposition attempt.
                except Exception:  # Error during transposition (e.g., parsing transposed pitch).
                    is_transposition_valid = False
                    break

            if is_transposition_valid:
                return temp_transposed_motif  # Return the successfully transposed motif.

        elif rand_variation_type < 0.9:  # Rhythmic variation (30% chance: 0.6 to 0.9)
            # Alter durations of some notes in the motif.
            for i in range(len(developed_motif)):
                pitch_str, current_duration = developed_motif[i]
                # More likely to vary rhythm of actual notes rather than rests.
                if pitch_str is not None and random.random() < 0.7:  # 70% chance to change duration of a note
                    # Choose a new duration from common rhythms, preferably different from the original.
                    # Avoid changing to the exact same duration.
                    new_duration_options = [d for d in POSSIBLE_DURATIONS if abs(d - current_duration) > 0.125]
                    if new_duration_options:
                        new_duration = random.choice(new_duration_options)
                        developed_motif[i] = (pitch_str, new_duration)
            return developed_motif

        # Else (10% chance): Return the original motif (no significant development applied this time).
        return developed_motif

    def generate_melody(self, total_notes_target: int,
                        ref_phrase_durations: Optional[List[float]] = None) -> MelodySequence:
        """
        Generates a complete melody, typically composed of several musical phrases.
        The melody aims to meet a target number of musical events (notes/rests) or
        to align with the structure defined by reference phrase durations if provided.

        Args:
            total_notes_target (int): An approximate target for the total number of musical events
                                     (notes/rests) in the melody. This is primarily used if
                                     `ref_phrase_durations` is None or insufficient.
            ref_phrase_durations (Optional[List[float]]): A list of target durations (in beats)
                                                          for each phrase. If provided, this guides
                                                          the number and length of phrases generated.

        Returns:
            MelodySequence: The generated melody as a list of (pitch, duration) tuples.
                            Returns an empty list if generation fails critically.
        """
        generated_melody: MelodySequence = []
        # Store already generated phrases; can be used for internal repetition/development.
        generated_phrases_history: List[MelodySequence] = []

        # --- Determine Phrase Structure ---
        # If reference phrase durations are provided, use them to define the number and target length of phrases.
        # Otherwise, estimate the number of phrases based on total_notes_target and a fallback notes-per-phrase count.
        num_phrases_to_generate = (len(ref_phrase_durations) if ref_phrase_durations
                                   else max(1, total_notes_target // NOTES_PER_PHRASE_FALLBACK))
        num_phrases_to_generate = max(1, num_phrases_to_generate)  # Ensure at least one phrase.

        # --- Contextual Variables for Melody Generation (persist across phrases) ---
        previous_sounded_pitch_obj: Optional[m21pitch.Pitch] = None  # Last actual pitch generated (not rest).
        consecutive_identical_pitch_count = 0  # Counter for MAX_CONSECUTIVE_REPEATED_PITCHES.
        last_melodic_leap_semitones = 0  # Interval of the last leap taken (for contour correction).

        # --- Loop to Generate Each Phrase ---
        for phrase_idx in range(num_phrases_to_generate):
            current_phrase_notes: MelodySequence = []  # Notes for the phrase currently being built.
            current_phrase_accumulated_duration = 0.0  # Duration of notes/rests in current_phrase_notes.

            # Determine target duration for this specific phrase.
            # Use reference if available, otherwise use default.
            target_duration_for_this_phrase = (ref_phrase_durations[phrase_idx]
                                               if ref_phrase_durations and phrase_idx < len(ref_phrase_durations)
                                               else self.target_phrase_duration_beats)

            # --- Optional: Internal Phrase Repetition/Development ---
            # With some probability, try to develop a previously generated phrase instead of creating a new one from scratch.
            # This can add thematic coherence to the melody.
            if generated_phrases_history and random.random() < INTERNAL_PHRASE_REPETITION_CHANCE:
                # Select a previous phrase that isn't too long to be a basis for development.
                # (Avoid developing an already very long phrase into something even longer if target is short).
                candidate_phrases_for_development = [
                    phr for phr in generated_phrases_history
                    if sum(n[1] for n in phr) < target_duration_for_this_phrase * 1.5  # Heuristic limit
                ]
                if candidate_phrases_for_development:
                    phrase_to_develop = random.choice(candidate_phrases_for_development)
                    developed_phrase_segment = self._apply_motif_development(phrase_to_develop)
                    developed_phrase_duration = sum(n[1] for n in developed_phrase_segment)

                    # If the developed phrase is valid and fits reasonably within the target duration, use it.
                    if developed_phrase_segment and developed_phrase_duration <= target_duration_for_this_phrase * 1.2:  # Allow slight overshoot
                        current_phrase_notes.extend(developed_phrase_segment)
                        current_phrase_accumulated_duration += developed_phrase_duration
                        # Update context (previous_sounded_pitch_obj, etc.) from the end of the developed phrase.
                        if developed_phrase_segment[-1][0] is not None:  # If last event was a note
                            try:
                                previous_sounded_pitch_obj = m21pitch.Pitch(developed_phrase_segment[-1][0])
                            except:
                                pass  # Keep previous_sounded_pitch_obj if parsing fails
                        # This loop iteration for generating more notes for this phrase will then be shorter or skipped.

            # --- Generate notes/rests for the current phrase until target duration is approached ---
            # Also limit by a maximum number of notes to prevent overly dense or fragmented phrases.
            # Estimate max notes: if shortest note is 0.25 beats (16th), phrase_duration / 0.25 gives max density.
            # Clamp this to a reasonable range.
            max_notes_for_this_phrase = int(
                target_duration_for_this_phrase / 0.25) if target_duration_for_this_phrase > 0 else NOTES_PER_PHRASE_FALLBACK * 2
            max_notes_for_this_phrase = max(MIN_NOTES_PER_PHRASE,
                                            min(max_notes_for_this_phrase, NOTES_PER_PHRASE_FALLBACK * 3))

            notes_generated_in_this_phrase_count = len(
                current_phrase_notes)  # Count notes already added from development

            # Loop to add notes/rests to the current phrase.
            # Stop if phrase duration is near target OR max notes for phrase is reached.
            # The 0.90 factor ensures we try to fill most of the target duration but leave a little room.
            while (current_phrase_accumulated_duration < target_duration_for_this_phrase * 0.90 and
                   notes_generated_in_this_phrase_count < max_notes_for_this_phrase):

                time_left_in_current_phrase = target_duration_for_this_phrase - current_phrase_accumulated_duration
                # Heuristic: if less than ~1 beat left, consider it "approaching end of phrase".
                is_approaching_phrase_end = time_left_in_current_phrase < 1.0  # Affects rest/duration choices

                # Get the next note/rest tuple using the core selection logic.
                next_pitch_str, next_duration = self._get_next_note_tuple(
                    previous_sounded_pitch_obj, last_melodic_leap_semitones,
                    time_left_in_current_phrase, is_approaching_phrase_end,
                    current_phrase_accumulated_duration, target_duration_for_this_phrase
                )

                # --- Handle Repeated Pitches (to avoid excessive monotony) ---
                if next_pitch_str is not None:  # If it's a note (not a rest)
                    if previous_sounded_pitch_obj and next_pitch_str == previous_sounded_pitch_obj.nameWithOctave:
                        consecutive_identical_pitch_count += 1
                        # If too many repetitions, try to force a change (with high probability).
                        if consecutive_identical_pitch_count >= MAX_CONSECUTIVE_REPEATED_PITCHES and random.random() < 0.95:
                            # Attempt to get a different pitch by temporarily modifying context for _get_next_note_tuple.
                            # Setting previous_sounded_pitch_obj to None makes _get_next_note_tuple less constrained.
                            forced_next_pitch_str, forced_next_duration = self._get_next_note_tuple(
                                None, last_melodic_leap_semitones,  # No immediate previous pitch context
                                time_left_in_current_phrase, is_approaching_phrase_end,
                                current_phrase_accumulated_duration, target_duration_for_this_phrase
                            )
                            # Use the forced change if it's different from the original repetition.
                            if forced_next_pitch_str != next_pitch_str:
                                next_pitch_str = forced_next_pitch_str
                                # next_duration = forced_next_duration # Optionally change duration too
                                consecutive_identical_pitch_count = 0  # Reset counter as pitch changed.
                            # If still same (e.g., limited choices), the original counter remains.
                    else:  # Pitch is different from previous, or previous was a rest.
                        consecutive_identical_pitch_count = 0  # Reset counter.
                else:  # If it's a rest, reset consecutive pitch repetition counter.
                    consecutive_identical_pitch_count = 0

                # Add the chosen note/rest to the current phrase.
                current_phrase_notes.append((next_pitch_str, next_duration))
                current_phrase_accumulated_duration += next_duration
                notes_generated_in_this_phrase_count += 1

                # --- Update Context for the Next Iteration ---
                if next_pitch_str is not None:  # If a note was generated
                    try:
                        current_pitch_obj = m21pitch.Pitch(next_pitch_str)
                        if previous_sounded_pitch_obj:  # Calculate leap from previous *sounding* pitch.
                            last_melodic_leap_semitones = interval.Interval(previous_sounded_pitch_obj,
                                                                            current_pitch_obj).semitones
                        else:  # No previous sounding pitch (start of melody or after a rest).
                            last_melodic_leap_semitones = 0
                        previous_sounded_pitch_obj = current_pitch_obj  # Update last sounded pitch.
                    except Exception:  # Error parsing pitch string.
                        last_melodic_leap_semitones = 0
                        # previous_sounded_pitch_obj remains as it was (or None).
                else:  # If a rest was generated, there's no new "leap" from it.
                    last_melodic_leap_semitones = 0
                    # previous_sounded_pitch_obj (the last *actual* pitch) remains unchanged for next note's reference.

            # --- Optional: Phrase Punctuation ---
            # With some probability, "punctuate" the phrase end, e.g., by lengthening the last note
            # if it's not already long, to give a sense of closure or pause.
            if current_phrase_notes and random.random() < PHRASE_PUNCTUATION_CHANCE:
                last_note_idx_in_phrase = len(current_phrase_notes) - 1
                last_event_pitch, last_event_duration = current_phrase_notes[last_note_idx_in_phrase]

                # Only punctuate if the last event is a note and not already very long.
                # Also, ensure the phrase has reached a substantial portion of its target duration.
                if last_event_pitch is not None and last_event_duration < 1.0 and \
                        current_phrase_accumulated_duration >= target_duration_for_this_phrase * 0.70:
                    # Choose a common longer duration for punctuation.
                    new_ending_duration = random.choice([d for d in POSSIBLE_DURATIONS if d >= 1.0] or [1.0])
                    # Adjust total phrase duration if changing last note's duration.
                    current_phrase_accumulated_duration = current_phrase_accumulated_duration - last_event_duration + new_ending_duration
                    current_phrase_notes[last_note_idx_in_phrase] = (last_event_pitch, new_ending_duration)

            # --- Finalize Phrase ---
            if current_phrase_notes:
                generated_melody.extend(current_phrase_notes)  # Add completed phrase to the main melody.
                generated_phrases_history.append(
                    list(current_phrase_notes))  # Store a copy for potential future development.

                # Update previous_sounded_pitch_obj from the end of this phrase for the start of the next.
                if current_phrase_notes[-1][0] is not None:  # If last event of phrase was a note
                    try:
                        previous_sounded_pitch_obj = m21pitch.Pitch(current_phrase_notes[-1][0])
                    except:
                        pass  # Keep previous if parsing fails; context carries over.
                # If phrase ended with rest, previous_sounded_pitch_obj (last actual sounded pitch) is already set.
            else:
                # If a phrase ended up empty (e.g., due to very short target duration or generation issues),
                # it's often best to stop further phrase generation for this melody to avoid cascading problems.
                print(
                    f"Warning (MelodyGenerator): Phrase {phrase_idx + 1} ended up empty. Stopping generation for this melody.")
                break  # Exit the phrase generation loop for this melody.

        # The generated melody might not exactly match `total_notes_target` due to duration-based
        # phrase generation and other dynamic factors. The fitness function will handle
        # discrepancies in length and duration when comparing against a reference.
        return generated_melody

    def mutate_melody(self, melody: MelodySequence, mutation_rate_param: float = MUTATION_RATE) -> MelodySequence:
        """
        Applies mutations to an existing melody. Mutations can include changing
        a note's pitch or duration, or changing a note to a rest (and vice-versa).
        The `mutation_rate_param` determines the probability for each component
        (pitch and duration of each event) to undergo a mutation attempt.

        Args:
            melody (MelodySequence): The melody (list of (pitch, duration) tuples) to mutate.
            mutation_rate_param (float): The probability for each gene (note/duration component)
                                         to undergo mutation. Defaults to MUTATION_RATE from constants.

        Returns:
            MelodySequence: The mutated melody. Returns the original melody if it's empty
                            or if no mutations occur.
        """
        if not melody: return []  # Cannot mutate an empty melody.

        mutated_melody = list(melody)  # Work on a copy to preserve the original.

        # Context for mutation, similar to generation, to guide pitch choices if a pitch is mutated.
        # This aims to make mutations more musically sensible rather than purely random.
        previous_pitch_obj_for_mutation_context: Optional[m21pitch.Pitch] = None
        last_leap_semitones_for_mutation_context = 0

        for i in range(len(mutated_melody)):
            original_pitch_str, original_duration = mutated_melody[i]
            # Initialize new values with original ones; they will be changed if mutation occurs.
            new_pitch_str, new_duration = original_pitch_str, float(original_duration)

            # --- Mutate Pitch (or change note/rest type) ---
            if random.random() < mutation_rate_param:  # Check if pitch component mutates
                if new_pitch_str is None:  # Currently a rest
                    # Chance for a rest to become a note.
                    # REST_PROBABILITY_MUTATION defines the chance of it *staying* a rest (or becoming one if it was a note).
                    # So, (1 - REST_PROBABILITY_MUTATION) is the chance of it becoming a note if it was a rest.
                    if random.random() > REST_PROBABILITY_MUTATION:
                        new_pitch_str = self._get_next_pitch_for_mutation(
                            previous_pitch_obj_for_mutation_context,
                            last_leap_semitones_for_mutation_context
                        )
                else:  # Currently a note
                    # Chance for a note to become a rest.
                    if random.random() < REST_PROBABILITY_MUTATION:
                        new_pitch_str = None
                    else:  # Mutate the existing pitch to a different one.
                        new_pitch_str = self._get_next_pitch_for_mutation(
                            previous_pitch_obj_for_mutation_context,
                            last_leap_semitones_for_mutation_context,
                            current_pitch_to_avoid=new_pitch_str  # Try to pick a *different* pitch
                        )

            # --- Mutate Duration ---
            if random.random() < mutation_rate_param:  # Check if duration component mutates
                new_duration = random.choice(POSSIBLE_DURATIONS)
                # If it became/is a rest, mutated rests are often shorter.
                if new_pitch_str is None and new_duration > 1.0:  # If it's a rest and new duration is long
                    new_duration = random.choice([d for d in POSSIBLE_DURATIONS if d <= 1.0])  # Prefer shorter rest

            mutated_melody[i] = (new_pitch_str, new_duration)  # Update the event in the melody

            # --- Special mutation for phrase endings (approximate heuristic) ---
            # This attempts to make phrase endings more musically conclusive by, e.g., lengthening the last note.
            # It's an approximation as true phrase boundaries are complex.
            # Check if the current note index suggests it might be the end of a typical phrase.
            is_approx_phrase_end = (i + 1) % NOTES_PER_PHRASE_FALLBACK == 0 and i > 0
            if is_approx_phrase_end and random.random() < PHRASE_END_MUTATION_CHANCE:
                current_p, current_d = mutated_melody[i]  # Get potentially mutated pitch/duration
                # If it's a note and currently short, potentially lengthen it for punctuation.
                if current_p is not None and current_d < 1.0:
                    # Choose a longer duration.
                    mutated_melody[i] = (current_p, random.choice([d for d in POSSIBLE_DURATIONS if d >= 1.0] or [1.0]))

            # --- Update Context for the Next Potential Pitch Mutation ---
            # This context helps make subsequent pitch mutations more musically coherent.
            try:
                if new_pitch_str is not None:  # If the current event is now a note
                    newly_set_pitch_obj = m21pitch.Pitch(new_pitch_str)
                    if previous_pitch_obj_for_mutation_context:
                        last_leap_semitones_for_mutation_context = interval.Interval(
                            previous_pitch_obj_for_mutation_context, newly_set_pitch_obj
                        ).semitones
                    else:  # No previous note context (e.g., start or after a rest)
                        last_leap_semitones_for_mutation_context = 0
                    previous_pitch_obj_for_mutation_context = newly_set_pitch_obj
                else:  # Current event is a rest
                    last_leap_semitones_for_mutation_context = 0
                    # previous_pitch_obj_for_mutation_context (last *sounded* pitch) remains for next note context.
            except Exception:  # Error parsing the new pitch string
                # Reset context if parsing fails to avoid issues with next iteration.
                last_leap_semitones_for_mutation_context = 0
                previous_pitch_obj_for_mutation_context = None

        return mutated_melody

    def _get_next_pitch_for_mutation(self,
                                     previous_pitch_obj: Optional[m21pitch.Pitch],
                                     last_leap_semitones: int,
                                     current_pitch_to_avoid: Optional[str] = None) -> str:
        """
        Internal helper function to select a new pitch during the mutation process.
        It's a simplified version of the pitch selection logic in `_get_next_note_tuple`,
        aimed at providing a musically plausible alternative pitch.

        Args:
            previous_pitch_obj: The music21.pitch.Pitch of the preceding note (if any).
            last_leap_semitones: The interval (semitones) of the last melodic leap.
            current_pitch_to_avoid: Optional. If provided, the selection will try not to pick
                                    this specific pitch string (e.g., the original pitch being mutated).

        Returns:
            str: The chosen pitch string (e.g., "C4"). Returns a fallback if no suitable pitch found.
        """
        if not self.allowed_pitches: return 'C4'  # Absolute fallback.

        # If no previous pitch context, choose somewhat randomly from weighted pitches, avoiding the current one.
        if previous_pitch_obj is None:
            candidate_pitches = [p for p in self.weighted_pitches if p != current_pitch_to_avoid]
            return random.choice(
                candidate_pitches if candidate_pitches else self.weighted_pitches)  # Fallback to any weighted if all are to_avoid

        # Generate candidates with weights based on proximity and contour, similar to _get_next_note_tuple.
        candidate_pitches_weighted: List[Tuple[str, int]] = []
        for p_str_candidate in self.allowed_pitches:
            if p_str_candidate == current_pitch_to_avoid:  # Skip if it's the pitch we want to change from.
                continue
            try:
                candidate_obj = m21pitch.Pitch(p_str_candidate)
                # Basic filter for very large jumps from previous pitch.
                if abs(candidate_obj.octave - previous_pitch_obj.octave) > 1 and \
                        abs(candidate_obj.ps - previous_pitch_obj.ps) > 12:  # More than an octave jump
                    continue

                interval_to_candidate = interval.Interval(previous_pitch_obj, candidate_obj)
                interval_semitones = interval_to_candidate.semitones
                abs_interval = abs(interval_semitones)

                weight = 0  # Base weight
                # Favor stepwise motion and small leaps.
                if abs_interval <= 2:
                    weight = 15  # High weight for steps
                elif abs_interval <= 4:
                    weight = 5  # Medium for small leaps (thirds)
                elif abs_interval <= ACCEPTABLE_LEAP_INTERVAL:
                    weight = 2  # Lower for larger, acceptable leaps (fifths)
                else:
                    weight = 1  # Minimal for very large leaps

                # Contour correction: if last leap was large, favor stepwise motion in opposite direction.
                if abs(last_leap_semitones) > ACCEPTABLE_LEAP_INTERVAL:
                    if abs_interval <= 2 and (
                            interval_semitones * last_leap_semitones < 0):  # Stepwise and opposite direction
                        weight *= 2  # Favor this corrective motion

                if weight > 0:
                    candidate_pitches_weighted.append((p_str_candidate, weight))
            except Exception:
                continue  # Skip problematic candidates.

        if not candidate_pitches_weighted:
            # Fallback: choose any allowed pitch that is not the one to avoid.
            fallback_options = [p for p in self.allowed_pitches if p != current_pitch_to_avoid]
            return random.choice(
                fallback_options if fallback_options else self.allowed_pitches)  # Ultimate fallback to any allowed

        # Choose from candidates based on their calculated weights.
        choices, weights_values = zip(*candidate_pitches_weighted)
        return random.choices(choices, weights=weights_values, k=1)[0]

# print(f"--- melody_generator.py (Comprehensive Comments, v. {__import__('datetime').date.today()}) loaded ---")
