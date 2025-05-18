# ga_components/genetic_algorithm_core.py
"""
Module: genetic_algorithm_core.py

Purpose:
This module implements the core logic of the genetic algorithm (GA) for evolving
musical melodies. It orchestrates the evolutionary process, including population
initialization, fitness evaluation, selection of parents, crossover (recombination),
and mutation to produce new generations of melodies. The fitness function aims
for a balance between adhering to a reference MIDI and general musicality.
"""

# Standard library imports
import random
from typing import List, Tuple, Optional, Dict, Any

# Third-party library imports (music21 for music theory operations)
from music21 import note, interval, pitch as m21pitch, key

# Local application/library specific imports
# Import constants that define GA parameters, fitness weights, and musical constraints.
from .music_constants import (
    POPULATION_SIZE, NOTES_PER_PHRASE_FALLBACK, MUTATION_RATE, CROSSOVER_RATE, TOURNAMENT_SIZE,
    MelodySequence, ReferencePhraseInfo, MelodyNote, MIN_NOTES_PER_PHRASE,
    WEIGHT_REF_PITCH_EXACT, WEIGHT_REF_RHYTHM_EXACT,
    WEIGHT_REF_EVENT_COUNT_MATCH, WEIGHT_REF_TOTAL_DURATION_MATCH,
    WEIGHT_REF_PHRASE_START_PITCH_MATCH, WEIGHT_REF_PHRASE_END_PITCH_MATCH,
    PENALTY_REF_PITCH_MISMATCH, PENALTY_REF_RHYTHM_MISMATCH,
    PENALTY_REF_NOTE_VS_REST_MISMATCH, PENALTY_OUT_OF_KEY,
    WEIGHT_SCALE_DEGREE_RESOLUTION, WEIGHT_STEPWISE_MOTION, MIN_STEPWISE_MOTION_RATIO,
    PENALTY_LARGE_LEAPS, WEIGHT_MELODY_RANGE_PENALTY,
    TARGET_MELODY_RANGE_SEMITONES, MAX_ALLOWED_MELODY_RANGE_SEMITONES,
    PENALTY_EXCESSIVE_REPETITION, MAX_CONSECUTIVE_REPEATED_PITCHES,
    ACCEPTABLE_LEAP_INTERVAL, WEIGHT_MELODIC_CONTOUR_SIMILARITY
)
# Import utility functions for music analysis (e.g., key parsing, phrase segmentation).
from .music_utils import MusicUtils
# Import the melody generator for creating and mutating musical sequences.
from .melody_generator import MelodyGenerator


def crossover(melody1: MelodySequence, melody2: MelodySequence) -> MelodySequence:
    """
    Performs single-point crossover between two parent melodies.
    A random crossover point is chosen, and segments from both parents are combined
    to create a new child melody.

    Args:
        melody1 (MelodySequence): The first parent melody.
        melody2 (MelodySequence): The second parent melody.

    Returns:
        MelodySequence: The resulting child melody. If crossover is not feasible
                        (e.g., parents are too short or invalid), a copy of one
                        of the parents or an empty list might be returned.
    """
    # Validate parent structures. Ensure they are lists of (pitch, duration) tuples.
    if not (isinstance(melody1, list) and isinstance(melody2, list) and
            melody1 and melody2 and  # Ensure they are not empty
            all(isinstance(n, tuple) and len(n) == 2 for n in melody1) and
            all(isinstance(n, tuple) and len(n) == 2 for n in melody2)):
        # Fallback: return a copy of a randomly chosen valid parent if one exists.
        valid_parents = [m[:] for m in [melody1, melody2] if isinstance(m, list) and m]
        return random.choice(valid_parents) if valid_parents else []

    len1, len2 = len(melody1), len(melody2)
    shorter_len = min(len1, len2)

    # Crossover requires at least two musical events in the shorter parent
    # to define a meaningful crossover point.
    if shorter_len < 2:
        return random.choice([melody1[:], melody2[:]])  # Return a copy of one parent

    # Choose a random crossover point (index).
    # The point is between the first and the last element of the shorter parent.
    point = random.randint(1, shorter_len - 1)

    # Create the child melody.
    child_melody = melody1[:point] + melody2[point:]
    return child_melody


def get_melodic_contour(melody_notes: List[Optional[m21pitch.Pitch]]) -> List[int]:
    """
    Calculates a simplified melodic contour for a sequence of pitches.
    -1 for downward movement, 1 for upward, 0 for same pitch or rest.

    Args:
        melody_notes (List[Optional[music21.pitch.Pitch]]): A list of pitch objects
                                                            (or None for rests).

    Returns:
        List[int]: A list representing the contour directions.
    """
    contour = []
    if len(melody_notes) < 2:
        return []  # Not enough notes for a contour segment

    for i in range(len(melody_notes) - 1):
        p1 = melody_notes[i]
        p2 = melody_notes[i + 1]

        if p1 is None or p2 is None:  # If either is a rest, no defined melodic movement
            contour.append(0)
        elif p2.ps > p1.ps:  # Pitch space value for comparison
            contour.append(1)  # Upward
        elif p2.ps < p1.ps:
            contour.append(-1)  # Downward
        else:
            contour.append(0)  # Same pitch
    return contour


def calculate_fitness_comprehensive(
        evolved_sequence: MelodySequence,
        reference_pitches_str: List[Optional[str]],  # String representations from reference
        reference_rhythms: List[float],
        reference_total_duration_secs: float,
        reference_phrases_data: Optional[List[ReferencePhraseInfo]],
        key_signature_str: str,
        tempo_bpm: int
) -> float:
    """
    Calculates a comprehensive fitness score for an evolved melody.
    This function balances strong adherence to a reference MIDI with various
    intrinsic musicality heuristics. A higher score indicates a "better" melody.

    Args:
        evolved_sequence: The generated melody to evaluate.
        reference_pitches_str: Pitch strings (or None for rests) from the reference MIDI.
        reference_rhythms: Durations (float) from the reference MIDI.
        reference_total_duration_secs: Total duration of the reference melody in seconds.
        reference_phrases_data: Analyzed phrase information from the reference MIDI.
        key_signature_str: The musical key context (e.g., "C major").
        tempo_bpm: The tempo in beats per minute, for duration calculations.

    Returns:
        float: The calculated fitness score.
    """
    raw_score = 0.0  # Initialize raw fitness score.
    num_events_evolved = len(evolved_sequence)

    # Heavily penalize empty or invalid melodies.
    if num_events_evolved == 0:
        return -1000.0

    # Obtain music21 Key object for music theory calculations.
    key_obj = MusicUtils.get_key_object(key_signature_str)

    # --- Category 1: Core Reference Matching (Highest Priority) ---
    num_events_reference = len(reference_pitches_str)
    if num_events_reference > 0:
        # 1.1. Event Count Matching: Score based on how closely the number of musical events
        # (notes/rests) in the evolved sequence matches the reference.
        event_count_diff = abs(num_events_evolved - num_events_reference)
        # Normalization: 1.0 for perfect match, decreasing towards 0 for larger differences.
        event_count_score_factor = 1.0 - (event_count_diff / float(max(1, num_events_reference)))
        raw_score += WEIGHT_REF_EVENT_COUNT_MATCH * event_count_score_factor

        # 1.2. Pitch and Rhythm Similarity (Event-by-Event Comparison)
        # Compare up to the length of the shorter sequence to avoid index errors.
        comparison_length = min(num_events_evolved, num_events_reference)
        for i in range(comparison_length):
            evolved_pitch_s, evolved_duration = evolved_sequence[i]
            ref_pitch_s, ref_duration = reference_pitches_str[i], reference_rhythms[i]

            # Pitch Matching:
            if evolved_pitch_s == ref_pitch_s:  # Exact match (pitch string or both are None for rests)
                raw_score += WEIGHT_REF_PITCH_EXACT
            elif (evolved_pitch_s is None) != (ref_pitch_s is None):  # Mismatch: one is note, other is rest
                raw_score += PENALTY_REF_NOTE_VS_REST_MISMATCH
            else:  # Both are notes, but their pitches differ
                raw_score += PENALTY_REF_PITCH_MISMATCH

            # Rhythm Matching:
            if abs(evolved_duration - ref_duration) < 0.01:  # Consider effectively exact for floats
                raw_score += WEIGHT_REF_RHYTHM_EXACT
            else:
                raw_score += PENALTY_REF_RHYTHM_MISMATCH

        # 1.3. Total Duration Matching (in seconds)
        # Calculate total duration of evolved sequence in quarter notes.
        evolved_total_duration_qn = sum(dur for _, dur in evolved_sequence)
        # Convert to seconds using the tempo.
        evolved_total_duration_secs = (evolved_total_duration_qn / tempo_bpm) * 60.0 if tempo_bpm > 0 else 0.0
        duration_diff_secs = abs(evolved_total_duration_secs - reference_total_duration_secs)
        # Normalize score: 1.0 for perfect match, decreasing. Avoid division by zero.
        max_possible_duration_diff_secs = max(1.0, reference_total_duration_secs)
        duration_score_factor = 1.0 - (duration_diff_secs / max_possible_duration_diff_secs)
        raw_score += WEIGHT_REF_TOTAL_DURATION_MATCH * duration_score_factor

        # --- Category 2: Phrase Boundary and Contour Matching (High Priority) ---
        if reference_phrases_data:
            # Analyze phrases in the evolved sequence for comparison.
            evolved_phrases_data = MusicUtils.analyze_reference_phrases(evolved_sequence, key_obj)
            num_phrases_to_compare = min(len(evolved_phrases_data), len(reference_phrases_data))

            ref_phrase_contours = []
            for ref_phr in reference_phrases_data[:num_phrases_to_compare]:
                ref_phrase_pitches_obj = [m21pitch.Pitch(p[0]) if p[0] is not None else None for p in ref_phr["notes"]]
                ref_phrase_contours.append(get_melodic_contour(ref_phrase_pitches_obj))

            for i in range(num_phrases_to_compare):
                evo_phrase = evolved_phrases_data[i]
                ref_phrase = reference_phrases_data[i]

                # 2.1. Phrase Start/End Pitch Matching:
                if evo_phrase["notes"] and ref_phrase["notes"]:  # Ensure both phrases have notes
                    # Start pitch
                    evo_start_pitch_s = evo_phrase["notes"][0][0]
                    ref_start_pitch_s = ref_phrase["notes"][0][0]
                    if evo_start_pitch_s == ref_start_pitch_s and evo_start_pitch_s is not None:
                        raw_score += WEIGHT_REF_PHRASE_START_PITCH_MATCH
                    # End pitch
                    evo_end_pitch_s = evo_phrase["notes"][-1][0]
                    ref_end_pitch_s = ref_phrase["notes"][-1][0]
                    if evo_end_pitch_s == ref_end_pitch_s and evo_end_pitch_s is not None:
                        raw_score += WEIGHT_REF_PHRASE_END_PITCH_MATCH

                # 2.2 Melodic Contour Similarity for corresponding phrases
                if i < len(ref_phrase_contours):  # Check if ref contour exists for this phrase
                    evo_phrase_pitches_obj = [m21pitch.Pitch(p[0]) if p[0] is not None else None for p in
                                              evo_phrase["notes"]]
                    evo_contour = get_melodic_contour(evo_phrase_pitches_obj)
                    ref_contour = ref_phrase_contours[i]

                    if evo_contour and ref_contour:
                        contour_comparison_len = min(len(evo_contour), len(ref_contour))
                        matches = sum(1 for k_c in range(contour_comparison_len) if
                                      evo_contour[k_c] == ref_contour[k_c] and ref_contour[
                                          k_c] != 0)  # Match non-static contour
                        if contour_comparison_len > 0:
                            similarity = matches / float(contour_comparison_len)
                            raw_score += WEIGHT_MELODIC_CONTOUR_SIMILARITY * similarity

    # --- Category 3 & 4: Intrinsic Musicality & Coherence (Moderate Influence) ---
    # These apply to the evolved sequence regardless of the reference, or to refine it.
    num_actual_notes_in_evolved = 0
    stepwise_movements = 0
    last_valid_pitch_obj: Optional[m21pitch.Pitch] = None  # Last successfully parsed pitch
    current_consecutive_repeats = 0
    max_consecutive_repeats_found = 0
    evolved_note_pitch_space_values = []  # For melodic range calculation

    for i in range(num_events_evolved):
        pitch_str_evolved, duration_evolved = evolved_sequence[i]

        if pitch_str_evolved is not None:  # Process only if it's a note
            num_actual_notes_in_evolved += 1
            try:
                current_pitch_obj_evolved = m21pitch.Pitch(pitch_str_evolved)
                evolved_note_pitch_space_values.append(current_pitch_obj_evolved.ps)

                # 3.1. In-Key Preference: Penalize notes not diatonic to the key.
                if not key_obj.isDiatonic(current_pitch_obj_evolved):
                    raw_score += PENALTY_OUT_OF_KEY

                if last_valid_pitch_obj:
                    # 3.2. Melodic Contour (Stepwise Motion vs. Large Leaps):
                    interval_between_notes = interval.Interval(last_valid_pitch_obj, current_pitch_obj_evolved)
                    abs_semitones = abs(interval_between_notes.semitones)
                    if 0 < abs_semitones <= 2:  # Stepwise (minor/major second)
                        stepwise_movements += 1
                    elif abs_semitones > ACCEPTABLE_LEAP_INTERVAL:  # Large leap
                        raw_score += PENALTY_LARGE_LEAPS

                    # 3.3. Tonal Resolution: Reward common melodic resolutions.
                    prev_degree = key_obj.getScaleDegreeFromPitch(last_valid_pitch_obj)
                    curr_degree = key_obj.getScaleDegreeFromPitch(current_pitch_obj_evolved)
                    if prev_degree and curr_degree:  # If both notes have defined scale degrees
                        # Example resolutions (can be expanded):
                        if (prev_degree == 7 and curr_degree == 1) or \
                                (prev_degree == 4 and curr_degree == 3 and key_obj.mode == 'major') or \
                                (prev_degree == 2 and curr_degree == 1):  # Common in major
                            raw_score += WEIGHT_SCALE_DEGREE_RESOLUTION

                    # 3.4. Consecutive Repeated Pitches:
                    if current_pitch_obj_evolved.nameWithOctave == last_valid_pitch_obj.nameWithOctave:
                        current_consecutive_repeats += 1
                    else:  # New pitch, reset counter after updating max
                        max_consecutive_repeats_found = max(max_consecutive_repeats_found, current_consecutive_repeats)
                        current_consecutive_repeats = 1  # Start new count for the current different pitch
                else:  # This is the first note encountered in a sequence (or after a rest)
                    current_consecutive_repeats = 1

                last_valid_pitch_obj = current_pitch_obj_evolved  # Update for next iteration
            except Exception:  # Error parsing the current pitch string
                raw_score += PENALTY_OUT_OF_KEY * 2  # Penalize unparseable notes more heavily
                last_valid_pitch_obj = None  # Reset context as current pitch is invalid
                max_consecutive_repeats_found = max(max_consecutive_repeats_found, current_consecutive_repeats)
                current_consecutive_repeats = 0  # Reset counter
        else:  # Current event is a rest
            last_valid_pitch_obj = None  # Reset pitch context after a rest
            max_consecutive_repeats_found = max(max_consecutive_repeats_found, current_consecutive_repeats)
            current_consecutive_repeats = 0  # Reset counter

    # Final update for max_consecutive_repeats after loop (in case melody ends with repeats)
    max_consecutive_repeats_found = max(max_consecutive_repeats_found, current_consecutive_repeats)

    # Apply score for overall stepwise motion ratio if there's more than one note.
    if num_actual_notes_in_evolved > 1:
        stepwise_ratio = stepwise_movements / float(num_actual_notes_in_evolved - 1)
        if stepwise_ratio >= MIN_STEPWISE_MOTION_RATIO:  # Reward if ratio meets minimum threshold
            raw_score += WEIGHT_STEPWISE_MOTION * stepwise_ratio  # Proportional reward

    # Apply penalty for excessive consecutive repeated pitches.
    if max_consecutive_repeats_found > MAX_CONSECUTIVE_REPEATED_PITCHES:
        # Penalty increases for each repeat beyond the allowed maximum.
        raw_score += (max_consecutive_repeats_found - MAX_CONSECUTIVE_REPEATED_PITCHES) * PENALTY_EXCESSIVE_REPETITION

    # 3.5. Melodic Range: Penalize melodies that are too wide or too static.
    if len(evolved_note_pitch_space_values) > 1:  # Need at least two notes to define a range
        min_ps = min(evolved_note_pitch_space_values)
        max_ps = max(evolved_note_pitch_space_values)
        melody_range_semitones = max_ps - min_ps

        if melody_range_semitones > MAX_ALLOWED_MELODY_RANGE_SEMITONES:
            # Stronger penalty if way out of the absolute maximum allowed range
            raw_score += WEIGHT_MELODY_RANGE_PENALTY * 1.5
        elif melody_range_semitones > TARGET_MELODY_RANGE_SEMITONES:
            # Normal penalty if over the preferred target range but within max allowed
            raw_score += WEIGHT_MELODY_RANGE_PENALTY
        # Optionally, could penalize very narrow ranges too if desired.

    return raw_score


class GeneticAlgorithm:
    """
    Manages the evolutionary process for generating musical melodies.
    This class handles the population of melodies, applies genetic operators
    (selection, crossover, mutation), and uses a fitness function to guide
    the evolution towards more musically desirable results.
    """

    def __init__(self, population_size: int,
                 melody_target_event_count: int,  # Target number of notes/rests for initial melodies
                 key_name_str: str,  # Musical key context (e.g., "C major")
                 tempo_bpm: int,  # Tempo in Beats Per Minute for duration calculations
                 ref_pitches_list: Optional[List[Optional[str]]] = None,  # Pitches from reference MIDI
                 ref_rhythms_list: Optional[List[float]] = None,  # Rhythms from reference MIDI
                 ref_phrases_data: Optional[List[ReferencePhraseInfo]] = None,  # Analyzed phrases from reference
                 ref_total_duration_secs: Optional[float] = None):  # Total duration of reference in seconds
        """
        Initializes the GeneticAlgorithm.
        """
        self.population_size: int = population_size
        self.melody_target_event_count: int = melody_target_event_count
        self.key_name: str = key_name_str
        self.tempo_bpm: int = tempo_bpm  # Store tempo for fitness calculations
        self.mutation_rate: float = MUTATION_RATE  # Use global constant

        # Store reference information if provided.
        self.ref_pitches: List[Optional[str]] = ref_pitches_list if ref_pitches_list else []
        self.ref_rhythms: List[float] = ref_rhythms_list if ref_rhythms_list else []
        self.ref_phrases_info: Optional[List[ReferencePhraseInfo]] = ref_phrases_data
        self.ref_total_duration_secs: float = ref_total_duration_secs if ref_total_duration_secs is not None else 0.0

        # Instantiate the melody generator for creating and mutating melodies.
        self.melody_generator: MelodyGenerator = MelodyGenerator(key_name_str)
        # Initialize the population of melodies.
        self.population: List[MelodySequence] = self._initialize_population()

    def _initialize_population(self) -> List[MelodySequence]:
        """
        Creates the initial population of melodies.
        Each melody is generated by the MelodyGenerator, guided by reference
        characteristics if available (event count, phrase durations).

        Returns:
            List[MelodySequence]: A list of generated melodies.
        """
        initial_pop: List[MelodySequence] = []

        # Determine the target number of events for initial melodies.
        # Prioritize reference length if available.
        current_target_event_count = self.melody_target_event_count
        if current_target_event_count <= 0:  # If not set or invalid
            if self.ref_pitches:  # Use length of reference pitch sequence
                current_target_event_count = len(self.ref_pitches)
            elif self.ref_phrases_info:  # Sum of notes in reference phrases
                current_target_event_count = sum(p['num_notes'] for p in self.ref_phrases_info)
            else:  # Absolute fallback if no reference info
                current_target_event_count = NOTES_PER_PHRASE_FALLBACK * 2  # e.g., 2 phrases

            if current_target_event_count <= 0:  # If still no valid target
                print(
                    "CRITICAL Error: Cannot determine target event count for GA population initialization. Returning empty.")
                return []
            # Update the instance variable if it was derived here
            self.melody_target_event_count = current_target_event_count

        # Extract reference phrase durations to guide the MelodyGenerator.
        ref_phrase_durations_for_gen: Optional[List[float]] = None
        if self.ref_phrases_info:
            ref_phrase_durations_for_gen = [p_info['duration_beats'] for p_info in self.ref_phrases_info]

        # Generate melodies until the population size is reached.
        for _ in range(self.population_size):  # Use self.population_size from constructor (via constants)
            melody = self.melody_generator.generate_melody(
                current_target_event_count,
                ref_phrase_durations_for_gen
            )
            # Add to population if valid (not empty and correct structure).
            if melody and isinstance(melody, list) and len(melody) > 0 and \
                    all(isinstance(n, tuple) and len(n) == 2 for n in melody):
                initial_pop.append(melody)

        # If generation was sparse, try to fill remaining spots.
        fill_attempts = 0
        max_fill_attempts = self.population_size  # Try up to population_size more times
        while len(initial_pop) < self.population_size and fill_attempts < max_fill_attempts:
            melody = self.melody_generator.generate_melody(current_target_event_count, ref_phrase_durations_for_gen)
            if melody and isinstance(melody, list) and len(melody) > 0:  # Basic validation
                initial_pop.append(melody)
            fill_attempts += 1

        if not initial_pop:  # If still completely empty after all attempts
            print("CRITICAL: Failed to initialize any valid individuals for GA population after fill attempts.")
            # As an absolute last resort, create a very simple placeholder melody to avoid crashes downstream.
            # This should ideally never be reached if MelodyGenerator is robust.
            placeholder_pitch = self.key_name.split(' ')[0] + "4"  # e.g. "C4"
            basic_melody = [(placeholder_pitch,
                             1.0)] * current_target_event_count if current_target_event_count > 0 else [
                (placeholder_pitch, 1.0)]
            initial_pop.append(basic_melody)
            print(f"Created a placeholder melody of {len(basic_melody)} events.")

        # Trim if over-generated (e.g. during fill attempts if initial was already full)
        return initial_pop[:self.population_size]

    def _calculate_fitness_for_individual(self, individual: MelodySequence) -> float:
        """
        Wrapper to call the main fitness calculation function for a single melody.
        Passes all necessary context (reference data, key, tempo) to the fitness function.

        Args:
            individual (MelodySequence): The melody to evaluate.

        Returns:
            float: The fitness score of the individual.
        """
        return calculate_fitness_comprehensive(  # Call the comprehensive fitness function
            individual, self.ref_pitches, self.ref_rhythms,
            self.ref_total_duration_secs, self.ref_phrases_info,
            self.key_name, self.tempo_bpm  # Pass tempo
        )

    def _select_parents(self) -> Tuple[MelodySequence, MelodySequence]:
        """
        Selects two parent melodies from the current population using Tournament Selection.
        A few individuals are chosen randomly, and the fittest among them is selected.
        This is done twice to get two parents.

        Returns:
            Tuple[MelodySequence, MelodySequence]: The two selected parent melodies.
        """
        if not self.population:  # Should be caught by run_one_generation, but defensive
            print("Warning: Parent selection called on empty GA population. Generating random new parents.")
            # Generate new random melodies if population is unexpectedly empty
            p1 = self.melody_generator.generate_melody(self.melody_target_event_count, None) or []
            p2 = self.melody_generator.generate_melody(self.melody_target_event_count, None) or []
            return (p1, p2)

        def tournament_select_one_parent() -> MelodySequence:
            """Helper function to select a single parent via tournament."""
            if not self.population:  # Should not happen if outer check passes
                return self.melody_generator.generate_melody(self.melody_target_event_count, None) or []

            # Ensure tournament size does not exceed current population size.
            actual_tournament_size = min(len(self.population), TOURNAMENT_SIZE)
            if actual_tournament_size == 0:  # Population became empty mid-process (highly unlikely)
                return self.melody_generator.generate_melody(self.melody_target_event_count, None) or []

            # Randomly select individuals for the tournament.
            tournament_candidates = random.sample(self.population, actual_tournament_size)
            # Select the individual with the highest fitness score from the tournament.
            winner = max(tournament_candidates, key=self._calculate_fitness_for_individual)
            return winner

        parent1 = tournament_select_one_parent()
        parent2 = tournament_select_one_parent()

        # Try to ensure parents are distinct, especially if population size allows.
        # This is not strictly required by tournament selection but can promote diversity.
        num_attempts_for_distinct_parent2 = 0
        while parent1 is parent2 and len(self.population) > 1 and num_attempts_for_distinct_parent2 < 5:
            parent2 = tournament_select_one_parent()  # Try to get a different second parent
            num_attempts_for_distinct_parent2 += 1

        return parent1, parent2

    def _crossover_melodies(self, parent1: MelodySequence, parent2: MelodySequence) -> MelodySequence:
        """
        Wrapper to call the main crossover function.
        Applies crossover with a certain probability (CROSSOVER_RATE). If crossover
        doesn't occur, one of the parents is returned as the child (asexual reproduction).

        Args:
            parent1 (MelodySequence): The first parent melody.
            parent2 (MelodySequence): The second parent melody.

        Returns:
            MelodySequence: The resulting child melody.
        """
        if random.random() < CROSSOVER_RATE:
            return crossover(parent1, parent2)  # Perform crossover
        else:
            # Asexual reproduction: randomly choose one parent to pass to the next generation (copied).
            return random.choice([parent1[:], parent2[:]])

    def _mutate_individual(self, melody: MelodySequence) -> MelodySequence:
        """
        Wrapper to call the mutation function from MelodyGenerator.
        The actual mutation logic, guided by musical rules, resides in `MelodyGenerator.mutate_melody`.

        Args:
            melody (MelodySequence): The melody to mutate.

        Returns:
            MelodySequence: The mutated melody.
        """
        return self.melody_generator.mutate_melody(melody, self.mutation_rate)

    def run_one_generation(self) -> List[MelodySequence]:
        """
        Executes one full generation cycle of the genetic algorithm:
        1. Validates current population.
        2. Creates offspring through selection, crossover, and mutation.
        3. Combines the current population with new offspring.
        4. Filters for valid individuals.
        5. Sorts the combined population by fitness (elitism: best individuals survive).
        6. Selects the top individuals to form the next generation's population.

        Returns:
            List[MelodySequence]: The new population after one generation.
                                   Returns an empty list if the population cannot be sustained.
        """
        # Ensure population is valid before starting a new generation.
        if not self.population and self.melody_target_event_count > 0:
            print("Warning: GA Population empty at start of generation. Attempting to re-initialize.")
            self.population = self._initialize_population()
            if not self.population:  # If re-initialization also fails
                print("CRITICAL Error: Failed to re-initialize GA population. Cannot run generation.")
                return []  # Cannot proceed if population is fundamentally broken.

        # --- Create Offspring Population ---
        offspring_population: List[MelodySequence] = []
        # Typically, generate a number of offspring equal to the population size
        # to maintain diversity and explore the solution space.
        num_offspring_to_generate = self.population_size

        for _ in range(num_offspring_to_generate):
            parent1, parent2 = self._select_parents()
            child = self._crossover_melodies(parent1, parent2)

            # Ensure child from crossover is valid before mutation.
            if child and isinstance(child, list) and len(child) > 0:
                mutated_child = self._mutate_individual(child)
                # Ensure mutated child is also valid.
                if mutated_child and isinstance(mutated_child, list) and len(mutated_child) > 0:
                    offspring_population.append(mutated_child)
                else:  # If mutation results in invalid child, consider adding the unmutated child.
                    offspring_population.append(child)  # Fallback to unmutated child
            else:
                # If crossover resulted in an invalid/empty child, generate a new random individual
                # to maintain population pressure.
                print("Warning: Invalid child from crossover. Generating a new random individual for offspring.")
                ref_phrase_durations_for_gen = [p['duration_beats'] for p in
                                                self.ref_phrases_info] if self.ref_phrases_info else None
                random_new_individual = self.melody_generator.generate_melody(
                    self.melody_target_event_count, ref_phrase_durations_for_gen
                )
                if random_new_individual:  # Ensure it's valid
                    offspring_population.append(random_new_individual)
                # If even random generation fails, this spot in offspring may remain empty or be handled by population trimming.

        # --- Combine, Validate, and Select Next Generation ---
        # Combine the current population with the generated offspring.
        combined_population = self.population + offspring_population

        # Filter out any invalid individuals that might have slipped through.
        valid_individuals_for_next_gen: List[MelodySequence] = []
        for individual in combined_population:
            if (isinstance(individual, list) and individual and  # Not empty
                    all(isinstance(item, tuple) and len(item) == 2 and \
                        (item[0] is None or isinstance(item[0], str)) and
                        isinstance(item[1], (float, int))
                        for item in individual)):
                valid_individuals_for_next_gen.append(individual)

        if not valid_individuals_for_next_gen:
            # Catastrophic failure: no valid individuals left. Try to recover.
            print(
                "CRITICAL Error: No valid individuals in combined GA population. Attempting to re-initialize population.")
            self.population = self._initialize_population()
            return self.population  # Return the newly initialized (hopefully valid) population.

        # Sort the valid combined population by fitness in descending order (higher score is better).
        valid_individuals_for_next_gen.sort(
            key=lambda ind: self._calculate_fitness_for_individual(ind),
            reverse=True  # Higher fitness scores are preferred
        )

        # Select the top individuals to form the next generation's population (elitism).
        self.population = valid_individuals_for_next_gen[:self.population_size]

        return self.population

# Informative print statement upon module loading.
# print(f"--- genetic_algorithm_core.py (Comprehensive Fitness, v. {__import__('datetime').date.today()}) loaded ---")
