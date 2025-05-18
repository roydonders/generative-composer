# ga_components/genetic_algorithm_core.py
"""
Module: genetic_algorithm_core.py

Purpose:
This module implements the core logic of the genetic algorithm (GA) for evolving
musical melodies. It orchestrates the evolutionary process, including population
initialization, fitness evaluation, selection of parents, crossover (recombination),
and mutation to produce new generations of melodies.

Key Components:
- GeneticAlgorithm class: Manages the GA lifecycle and operations.
- Fitness function: Evaluates the quality of melodies based on various musical criteria.
- Crossover function: Combines genetic material from two parent melodies.
"""

from music21 import note, interval, pitch as m21pitch, key, roman # roman unused here directly
import random
import numpy # For potential use in fitness or advanced selection, currently minimal.
from typing import List, Tuple, Optional, Dict, Any # Set was unused

# Import constants and type definitions from the music_constants module.
# This ensures consistency in parameters and data structures.
from .music_constants import *
# Import utility functions and the melody generator.
from .music_utils import MusicUtils
from .melody_generator import MelodyGenerator


def crossover(melody1: MelodySequence, melody2: MelodySequence) -> MelodySequence:
    """
    Performs single-point crossover between two parent melodies.
    The crossover point is chosen randomly based on the number of notes/events.
    The resulting child melody may have a variable length, inheriting segments
    from both parents.

    Args:
        melody1: The first parent melody (list of (pitch, duration) tuples).
        melody2: The second parent melody.

    Returns:
        A new MelodySequence representing the child melody. Returns one of the
        parents if crossover is not feasible (e.g., parents are too short).
    """
    # Basic validation of parent melodies.
    if not (isinstance(melody1, list) and isinstance(melody2, list) and
            melody1 and melody2 and # Ensure they are not empty
            all(isinstance(n, tuple) and len(n) == 2 for n in melody1) and
            all(isinstance(n, tuple) and len(n) == 2 for n in melody2)):
        # If validation fails, return a copy of a random valid parent or an empty list.
        valid_parents = [m[:] for m in [melody1, melody2] if isinstance(m, list) and m]
        return random.choice(valid_parents) if valid_parents else []

    len1, len2 = len(melody1), len(melody2)
    shorter_len = min(len1, len2)

    # Crossover requires at least two notes/events in the shorter parent to define a segment.
    if shorter_len < 2:
        return random.choice([melody1[:], melody2[:]]) # Return a copy of one parent

    # Choose a random crossover point (index of a note/event).
    # The point is between the first and the last element of the shorter parent.
    point = random.randint(1, shorter_len - 1)

    # Create the child melody by combining segments from both parents.
    # Child takes the first part of melody1 (up to the point) and the second part of melody2 (from the point).
    child_melody = melody1[:point] + melody2[point:]

    # The length of the child melody can vary, which is handled by the fitness function.
    return child_melody


def calculate_fitness(
    evolved_sequence: MelodySequence,
    reference_pitches: List[Optional[str]], # Note: Pitches can be None for rests.
    reference_rhythms: List[float],
    key_signature_str: str,
    reference_phrases_info: Optional[List[ReferencePhraseInfo]] = None # List of TypedDicts
) -> float:
    """
    Calculates the fitness score for a given evolved melody.
    The fitness score quantifies how "good" the melody is based on various musical criteria,
    including similarity to a reference MIDI (if provided) and intrinsic musicality heuristics.
    A lower score generally indicates better fitness (as it's often used in minimization contexts,
    but here a higher raw_score is better, so we might return -raw_score or adjust selection).
    This implementation aims for a higher raw_score = better, and the GA will sort to keep highest.

    Args:
        evolved_sequence: The generated melody to evaluate.
        reference_pitches: List of pitch strings (or None for rests) from the reference MIDI.
        reference_rhythms: List of durations (float) from the reference MIDI.
        key_signature_str: The musical key of the reference/target context.
        reference_phrases_info: Optional list of analyzed phrase information from the reference MIDI.

    Returns:
        float: The calculated fitness score. Higher is better.
               Returns negative infinity for an empty or invalid sequence.
    """
    raw_score = 0.0  # Initialize raw fitness score.
    num_events_evolved = len(evolved_sequence)

    if num_events_evolved == 0:
        return float('-inf') # An empty melody has the worst possible fitness.

    # --- Setup: Key and Tonal Context ---
    key_obj = MusicUtils.get_key_object(key_signature_str)
    # tonal_hierarchy = MusicUtils.get_tonal_hierarchy(key_obj) # For checking resolutions, etc.
    # scale_pitch_classes = {p.name for p in key_obj.getPitches()} # Pitch classes in the key

    # --- Analyze phrases in the evolved sequence for comparison and intrinsic evaluation ---
    # This uses the same logic as analyzing reference phrases.
    evolved_phrases_info: List[ReferencePhraseInfo] = MusicUtils.analyze_reference_phrases(evolved_sequence, key_obj)


    # --- I. Reference-Based Scoring (if reference MIDI data is available) ---
    num_events_reference = len(reference_pitches)
    if num_events_reference > 0:
        # 1. Length Mismatch Penalty: Compare number of musical events (notes/rests).
        # Penalize deviation from the reference length.
        length_mismatch = abs(num_events_reference - num_events_evolved)
        # Penalty increases with the square of the mismatch to strongly discourage large deviations.
        # Max possible mismatch is num_events_reference (if evolved is empty) or num_events_evolved (if ref is empty)
        # Normalizing factor could be num_events_reference. Let's use a simple factor.
        raw_score -= length_mismatch * 2.0 # Moderate penalty per event difference.

        # 2. Pitch and Rhythm Similarity (event by event comparison)
        # Compare up to the length of the shorter sequence to avoid index errors.
        comparison_length = min(num_events_evolved, num_events_reference)
        reference_rhythms_float = [float(r) for r in reference_rhythms] # Ensure floats

        for i in range(comparison_length):
            evolved_pitch_str, evolved_duration = evolved_sequence[i]
            ref_pitch_str, ref_rhythm_float = reference_pitches[i], reference_rhythms_float[i]

            # a. Pitch Matching (handles rests: None == None is True)
            if evolved_pitch_str == ref_pitch_str: # Exact match (pitch or rest type)
                raw_score += WEIGHT_REF_PITCH_EXACT
            elif evolved_pitch_str is not None and ref_pitch_str is not None: # Both are notes, but different
                try:
                    p_evolved = m21pitch.Pitch(evolved_pitch_str)
                    p_ref = m21pitch.Pitch(ref_pitch_str)
                    semitone_difference = abs(interval.Interval(p_evolved, p_ref).semitones)
                    if semitone_difference <= 2: # Close match (e.g., within a whole step)
                        raw_score += WEIGHT_REF_PITCH_CLOSE
                    else: # Significantly different pitches
                        # Penalty proportional to how far off it is, normalized by octave.
                        raw_score -= (semitone_difference / 12.0) * WEIGHT_REF_PITCH_PENALTY_FACTOR
                except Exception: # Error parsing one of the pitches
                    raw_score -= WEIGHT_REF_PITCH_PENALTY_FACTOR # Penalize if unparseable
            else:  # Mismatch between a note and a rest
                raw_score -= WEIGHT_REF_PITCH_EXACT * 0.75 # Significant penalty for note/rest type mismatch

            # b. Rhythm Matching
            rhythm_difference = abs(evolved_duration - ref_rhythm_float)
            if rhythm_difference < 0.01: # Effectively exact match for float durations
                raw_score += WEIGHT_REF_RHYTHM_EXACT
            # Consider "close" if within a certain percentage of the reference duration.
            elif ref_rhythm_float > 0 and rhythm_difference < (ref_rhythm_float * 0.25): # Within 25%
                raw_score += WEIGHT_REF_RHYTHM_CLOSE
            else: # Significantly different rhythms
                raw_score -= WEIGHT_REF_RHYTHM_EXACT * 0.5 # Penalty for rhythm mismatch


        # 3. Phrase Structure and Character Matching (if reference phrases were analyzed)
        if reference_phrases_info and evolved_phrases_info:
            num_phrases_to_compare = min(len(reference_phrases_info), len(evolved_phrases_info))
            for i in range(num_phrases_to_compare):
                ref_phrase = reference_phrases_info[i]
                evo_phrase = evolved_phrases_info[i]

                # a. Matching Phrase Character (Question/Answer dynamics)
                ref_char = ref_phrase.get("character", "neutral")
                evo_char = evo_phrase.get("character", "neutral")
                if ref_char == evo_char and ref_char != "neutral": # Matched non-neutral character
                    raw_score += WEIGHT_REF_QA_CHARACTER_MATCH
                elif ref_char != "neutral" and evo_char != "neutral" and ref_char != evo_char: # Mismatched Q/A
                    raw_score -= WEIGHT_REF_QA_CHARACTER_MATCH * 0.75 # Penalize mismatch

                # b. Note Similarity within corresponding phrases (if characters match or are both neutral)
                # This rewards melodies that use similar notes even if overall Q/A doesn't match perfectly.
                ref_phrase_notes = ref_phrase.get("notes", [])
                evo_phrase_notes = evo_phrase.get("notes", [])
                if ref_phrase_notes and evo_phrase_notes:
                    notes_to_compare_in_phrase = min(len(ref_phrase_notes), len(evo_phrase_notes))
                    phrase_note_matches = 0
                    for k in range(notes_to_compare_in_phrase):
                        if ref_phrase_notes[k][0] == evo_phrase_notes[k][0]: # Pitch match
                            phrase_note_matches += 1
                    similarity_ratio = phrase_note_matches / notes_to_compare_in_phrase if notes_to_compare_in_phrase > 0 else 0
                    raw_score += similarity_ratio * WEIGHT_REF_QA_PHRASE_NOTE_SIMILARITY


    # --- II. Intrinsic Musicality Heuristics (applied to the evolved sequence) ---
    # These scores evaluate the melody's quality independent of a reference.

    # 1. Phrase Endings: Encourage phrases to end on longer notes or rests for punctuation.
    #    Penalize overly long notes appearing mid-phrase that might halt momentum.
    for evo_phr_info in evolved_phrases_info:
        if evo_phr_info["notes"]: # Check if phrase has notes
            _pitch_last, duration_last = evo_phr_info["notes"][-1] # Last event of the phrase
            if duration_last >= 1.5: # e.g., dotted quarter or longer at phrase end
                raw_score += WEIGHT_PHRASE_END_LONG_NOTE_OR_REST

            # Penalize very long notes occurring *before* the end of the phrase.
            for k_note_idx in range(len(evo_phr_info["notes"]) - 1): # Iterate up to penultimate note
                _p_mid, dur_mid = evo_phr_info["notes"][k_note_idx]
                if dur_mid >= 2.0: # e.g., Half note or longer mid-phrase is often too static.
                    raw_score += WEIGHT_PENALTY_MID_PHRASE_LONG_NOTE # Note: this weight is negative in constants.
                                                                    # Or change constant to positive and subtract here.
                                                                    # Assuming positive weight from constants:
                                                                    # raw_score -= WEIGHT_PENALTY_MID_PHRASE_LONG_NOTE

    # 2. Melodic Range: Prefer melodies that stay within a comfortable "singing" range.
    #    Penalize melodies that are too wide or too static (narrow).
    note_pitches_ps = [] # List to store pitch space values (MIDI note numbers)
    for p_str, _dur in evolved_sequence:
        if p_str is not None:
            try: note_pitches_ps.append(m21pitch.Pitch(p_str).ps)
            except: pass # Ignore unparseable pitches for range calculation

    if len(note_pitches_ps) > 1:
        min_ps, max_ps = min(note_pitches_ps), max(note_pitches_ps)
        melody_range_semitones = max_ps - min_ps
        if melody_range_semitones > MAX_ALLOWED_MELODY_RANGE_SEMITONES:
            raw_score -= (melody_range_semitones - MAX_ALLOWED_MELODY_RANGE_SEMITONES) * 0.5 # Penalty for exceeding hard max
        elif melody_range_semitones > TARGET_MELODY_RANGE_SEMITONES:
            # Softer penalty for exceeding preferred target range but still within max allowed.
            raw_score += WEIGHT_MELODY_RANGE_PENALTY # This weight is negative.
                                                    # raw_score -= (melody_range_semitones - TARGET_MELODY_RANGE_SEMITONES) * abs(WEIGHT_MELODY_RANGE_PENALTY)

    # 3. In-Key Preference & Tonal Resolution:
    #    Reward notes that are diatonic to the key. Penalize out-of-key notes.
    #    Reward common tonal resolutions (e.g., leading tone to tonic).
    num_notes_in_sequence = 0
    num_stepwise_moves = 0
    previous_pitch_obj: Optional[m21pitch.Pitch] = None

    for i in range(num_events_evolved):
        pitch_s, duration = evolved_sequence[i]
        if pitch_s is None: # Skip rests for these checks
            previous_pitch_obj = None # Reset context after a rest
            continue
        num_notes_in_sequence +=1
        try:
            current_pitch_obj = m21pitch.Pitch(pitch_s)
            # a. In-Key Check
            if not key_obj.isDiatonic(current_pitch_obj): # Check if pitch is in the current key
                raw_score += WEIGHT_OUT_OF_KEY_PENALTY # This weight is negative.
            else: # If in key, small bonus
                 raw_score += abs(WEIGHT_OUT_OF_KEY_PENALTY) * 0.1 # Small bonus for being in key

            # b. Tonal Resolution & Stepwise Motion (requires previous note)
            if previous_pitch_obj:
                # Stepwise motion check
                interval_semitones = abs(interval.Interval(previous_pitch_obj, current_pitch_obj).semitones)
                if interval_semitones > 0 and interval_semitones <= 2: # Melodic interval of 1 or 2 semitones
                    num_stepwise_moves += 1
                elif interval_semitones > ACCEPTABLE_LEAP_INTERVAL: # Check for large leaps
                    # Penalize large leaps, more so if not harmonically justified (complex to check here simply)
                    # WEIGHT_LARGE_LEAP_PENALTY is positive, so subtract.
                    raw_score -= (interval_semitones - ACCEPTABLE_LEAP_INTERVAL) * WEIGHT_LARGE_LEAP_PENALTY * 0.1


                # Tonal Resolution Check (e.g. 7->1, 4->3)
                # This is a simplified check. MusicUtils.get_tonal_function can be used.
                prev_degree = key_obj.getScaleDegreeFromPitch(previous_pitch_obj)
                curr_degree = key_obj.getScaleDegreeFromPitch(current_pitch_obj)
                if prev_degree and curr_degree:
                    if (prev_degree == 7 and curr_degree == 1) or \
                       (prev_degree == 4 and curr_degree == 3 and key_obj.mode == 'major') or \
                       (prev_degree == 2 and curr_degree == 1):
                        raw_score += WEIGHT_SCALE_DEGREE_RESOLUTION
            previous_pitch_obj = current_pitch_obj
        except Exception:
            raw_score += WEIGHT_OUT_OF_KEY_PENALTY # Penalize if pitch is unparseable

    # Reward overall stepwise motion ratio
    if num_notes_in_sequence > 1:
        stepwise_ratio = num_stepwise_moves / (num_notes_in_sequence -1)
        if stepwise_ratio >= MIN_STEPWISE_MOTION_RATIO:
            raw_score += WEIGHT_STEPWISE_MOTION * stepwise_ratio


    # 4. Consecutive Repeated Pitches: Penalize excessive repetition of the same pitch.
    max_consecutive_repeats_found = 0
    current_consecutive_repeats = 0
    for i in range(num_events_evolved):
        current_event_pitch = evolved_sequence[i][0]
        if i > 0 and current_event_pitch is not None and current_event_pitch == evolved_sequence[i-1][0]:
            current_consecutive_repeats += 1
        else:
            max_consecutive_repeats_found = max(max_consecutive_repeats_found, current_consecutive_repeats)
            current_consecutive_repeats = 1 if current_event_pitch is not None else 0
    max_consecutive_repeats_found = max(max_consecutive_repeats_found, current_consecutive_repeats)

    if max_consecutive_repeats_found > MAX_CONSECUTIVE_REPEATED_PITCHES:
        # WEIGHT_EXCESSIVE_REPEATED_PITCH_PENALTY is positive, so subtract.
        raw_score -= (max_consecutive_repeats_found - MAX_CONSECUTIVE_REPEATED_PITCHES) * WEIGHT_EXCESSIVE_REPEATED_PITCH_PENALTY

    # 5. Internal Phrase/Motif Repetition & Development (Coherence)
    # Rewards if phrases within the evolved melody show some similarity (motivic development).
    # This uses the analyzed `evolved_phrases_info`.
    if len(evolved_phrases_info) > 1:
        for i in range(len(evolved_phrases_info)):
            phrase1_notes = evolved_phrases_info[i]["notes"]
            for j in range(i + 1, len(evolved_phrases_info)): # Compare with subsequent phrases
                phrase2_notes = evolved_phrases_info[j]["notes"]
                # Simple comparison: if two phrases (of similar length) have a high degree of pitch match.
                if len(phrase1_notes) == len(phrase2_notes) and len(phrase1_notes) > MIN_NOTES_PER_PHRASE:
                    matches = sum(1 for k_idx in range(len(phrase1_notes))
                                  if phrase1_notes[k_idx][0] == phrase2_notes[k_idx][0] and \
                                     phrase1_notes[k_idx][0] is not None) # Count matching non-rest pitches
                    match_ratio = matches / len(phrase1_notes)
                    if match_ratio > 0.6: # If >60% pitch match for same-length phrases
                        raw_score += WEIGHT_INTERNAL_PHRASE_REPETITION * match_ratio
                        # Could add more sophisticated checks (transposition, rhythmic variation etc.)

    return raw_score # Higher raw_score should mean better fitness for sorting.


class GeneticAlgorithm:
    """
    Manages the evolutionary process for generating musical melodies.
    This class handles the population of melodies, applies genetic operators
    (selection, crossover, mutation), and uses a fitness function to guide
    the evolution towards more musically desirable results.
    """

    def __init__(self, population_size: int,
                 melody_target_event_count: int, # Target number of notes/rests for generated melodies
                 key_name_str: str,
                 # mutation_rate_param: float = MUTATION_RATE, # Use MUTATION_RATE from constants directly
                 ref_pitches_list: Optional[List[Optional[str]]] = None, # Renamed for clarity
                 ref_rhythms_list: Optional[List[float]] = None,       # Renamed for clarity
                 ref_phrases_data: Optional[List[ReferencePhraseInfo]] = None): # Renamed for clarity
        """
        Initializes the GeneticAlgorithm.

        Args:
            population_size (int): The number of melodies in each generation.
            melody_target_event_count (int): The target number of musical events (notes/rests)
                                            for initially generated melodies.
            key_name_str (str): The musical key context for melody generation.
            ref_pitches_list (Optional[List[Optional[str]]]): Pitches from a reference MIDI.
            ref_rhythms_list (Optional[List[float]]): Rhythms from a reference MIDI.
            ref_phrases_data (Optional[List[ReferencePhraseInfo]]): Analyzed phrase info from reference.
        """
        self.population_size: int = population_size
        self.melody_target_event_count: int = melody_target_event_count
        self.key_name: str = key_name_str
        self.mutation_rate: float = MUTATION_RATE # Use the global constant

        # Store reference information if provided.
        self.ref_pitches: List[Optional[str]] = ref_pitches_list if ref_pitches_list else []
        self.ref_rhythms: List[float] = ref_rhythms_list if ref_rhythms_list else []
        self.ref_phrases_info: Optional[List[ReferencePhraseInfo]] = ref_phrases_data # Can be None

        # Instantiate the melody generator for creating and mutating melodies.
        self.melody_generator: MelodyGenerator = MelodyGenerator(key_name_str)
        # Initialize the population of melodies.
        self.population: List[MelodySequence] = self._initialize_population()

    def _initialize_population(self) -> List[MelodySequence]:
        """
        Creates the initial population of melodies.
        Each melody is generated randomly by the MelodyGenerator.

        Returns:
            List[MelodySequence]: A list of generated melodies.
        """
        initial_pop: List[MelodySequence] = []
        if self.melody_target_event_count <= 0:
            print("Warning: melody_target_event_count is zero or negative. Cannot initialize population.")
            return []

        # Extract reference phrase durations if available to guide generation length.
        # This helps generate initial melodies that are structurally similar to the reference.
        ref_durations_for_gen: Optional[List[float]] = None
        if self.ref_phrases_info:
            ref_durations_for_gen = [p_info['duration_beats'] for p_info in self.ref_phrases_info]

        # Generate melodies until the population size is reached.
        # Includes a limit on attempts to prevent infinite loops if generation is problematic.
        max_init_attempts = self.population_size * 3 # Allow more attempts
        attempts = 0
        while len(initial_pop) < self.population_size and attempts < max_init_attempts :
            # Generate a melody using the target event count and reference phrase durations.
            melody = self.melody_generator.generate_melody(
                self.melody_target_event_count,
                ref_durations_for_gen
            )
            # Add to population if valid (not empty and correct structure).
            if melody and isinstance(melody, list) and len(melody) > 0 and \
               all(isinstance(n, tuple) and len(n)==2 for n in melody):
                initial_pop.append(melody)
            attempts += 1

        if len(initial_pop) < self.population_size:
            print(f"Warning: Could only initialize {len(initial_pop)}/{self.population_size} valid individuals.")
        return initial_pop

    def _calculate_fitness_for_individual(self, individual: MelodySequence) -> float:
        """
        Wrapper to call the main fitness calculation function for a single melody.
        Passes all necessary context (reference data, key) to the fitness function.

        Args:
            individual: The MelodySequence (melody) to evaluate.

        Returns:
            float: The fitness score of the individual.
        """
        return calculate_fitness(
            individual, self.ref_pitches, self.ref_rhythms, self.key_name, self.ref_phrases_info
        )

    def _select_parents(self) -> Tuple[MelodySequence, MelodySequence]:
        """
        Selects two parent melodies from the current population for reproduction.
        This implementation uses tournament selection: a few individuals are chosen
        randomly, and the one with the best fitness among them is selected as a parent.
        This is done twice to get two parents.

        Returns:
            Tuple[MelodySequence, MelodySequence]: The two selected parent melodies.
                                                   Returns random new melodies if population is empty.
        """
        if not self.population: # Should ideally not happen if initialized.
            print("Warning: Parent selection called on empty population. Generating random parents.")
            p1 = self.melody_generator.generate_melody(self.melody_target_event_count, None)
            p2 = self.melody_generator.generate_melody(self.melody_target_event_count, None)
            return (p1 if p1 else [], p2 if p2 else []) # Ensure non-None return

        # Tournament Selection
        def tournament_select_one() -> MelodySequence:
            if len(self.population) < TOURNAMENT_SIZE:
                # If population smaller than tournament size, just pick the best of the population.
                return max(self.population, key=self._calculate_fitness_for_individual)

            tournament_candidates = random.sample(self.population, TOURNAMENT_SIZE)
            # Select the individual with the highest fitness score from the tournament.
            winner = max(tournament_candidates, key=self._calculate_fitness_for_individual)
            return winner

        parent1 = tournament_select_one()
        parent2 = tournament_select_one()

        # Ensure parents are distinct if possible, though not strictly required by tournament selection.
        # For very small populations, they might be the same.
        num_attempts_distinct_parent2 = 0
        while parent1 is parent2 and len(self.population) > 1 and num_attempts_distinct_parent2 < 5:
            parent2 = tournament_select_one() # Try to get a different second parent
            num_attempts_distinct_parent2 +=1

        return parent1, parent2


    def _crossover_melodies(self, parent1: MelodySequence, parent2: MelodySequence) -> MelodySequence:
        """
        Wrapper to call the main crossover function.
        Applies crossover with a certain probability (CROSSOVER_RATE). If crossover
        doesn't occur, one of the parents is returned as the child (asexual reproduction).

        Args:
            parent1: The first parent melody.
            parent2: The second parent melody.

        Returns:
            MelodySequence: The resulting child melody (either from crossover or a parent).
        """
        if random.random() < CROSSOVER_RATE:
            return crossover(parent1, parent2)
        else:
            # Asexual reproduction: randomly choose one parent to pass to the next generation.
            return random.choice([parent1[:], parent2[:]]) # Return a copy


    def _mutate_individual(self, melody: MelodySequence) -> MelodySequence:
        """
        Wrapper to call the mutation function from MelodyGenerator.
        The actual mutation logic resides in MelodyGenerator.mutate_melody.

        Args:
            melody: The MelodySequence to mutate.

        Returns:
            MelodySequence: The mutated melody.
        """
        return self.melody_generator.mutate_melody(melody, self.mutation_rate)


    def run_one_generation(self) -> List[MelodySequence]:
        """
        Executes one full generation cycle of the genetic algorithm:
        1. Selects parents from the current population.
        2. Creates offspring through crossover and mutation.
        3. Combines the current population with the new offspring.
        4. Sorts the combined population by fitness.
        5. Selects the fittest individuals to form the next generation's population (elitism).

        Returns:
            List[MelodySequence]: The new population after one generation.
                                   Returns an empty list if the population cannot be sustained.
        """
        # Ensure population is valid before starting.
        if not self.population and self.melody_target_event_count > 0:
            print("Warning: Population empty at start of generation. Re-initializing.")
            self.population = self._initialize_population()
            if not self.population:
                print("Error: Failed to re-initialize population. Cannot run generation.")
                return [] # Cannot proceed if population is fundamentally broken.

        # --- Create Offspring ---
        offspring_population: List[MelodySequence] = []
        # Generate a number of offspring typically equal to the population size to maintain diversity.
        # Some GAs use fewer offspring if elitism is strong.
        num_offspring_to_generate = self.population_size

        for _ in range(num_offspring_to_generate):
            parent1, parent2 = self._select_parents()
            child = self._crossover_melodies(parent1, parent2)

            # Basic validation for the child from crossover
            if child and isinstance(child, list) and len(child) > 0 and \
               all(isinstance(n, tuple) and len(n) == 2 for n in child):
                mutated_child = self._mutate_individual(child)
                # Validate mutated child
                if mutated_child and isinstance(mutated_child, list) and len(mutated_child) > 0:
                    offspring_population.append(mutated_child)
                else: # If mutation results in invalid child, add original child (less ideal) or a new random one
                    offspring_population.append(child) # Fallback to unmutated child
            else: # If crossover resulted in invalid child, generate a new random individual
                print("Warning: Invalid child from crossover. Generating a new random individual for offspring.")
                random_new_individual = self.melody_generator.generate_melody(
                    self.melody_target_event_count,
                    [p_info['duration_beats'] for p_info in self.ref_phrases_info] if self.ref_phrases_info else None
                )
                if random_new_individual: # Ensure it's valid
                    offspring_population.append(random_new_individual)
                # If even random generation fails, this spot in offspring remains empty for now.


        # --- Combine and Select Next Generation ---
        # Combine the current population with the generated offspring.
        # This allows for elitism, where the best individuals from the previous generation can survive.
        combined_population = self.population + offspring_population

        # Filter out any invalid individuals that might have slipped through or resulted from problematic operations.
        valid_individuals_for_next_gen: List[MelodySequence] = []
        for individual in combined_population:
            if (isinstance(individual, list) and individual and # Not empty
                all(isinstance(item, tuple) and len(item) == 2 and \
                    (item[0] is None or isinstance(item[0], str)) and  # Pitch is None (rest) or string
                    isinstance(item[1], (float, int)) # Duration is number
                    for item in individual)):
                valid_individuals_for_next_gen.append(individual)

        if not valid_individuals_for_next_gen:
            # Catastrophic failure: no valid individuals left. Try to recover by re-initializing.
            print("Critical Error: No valid individuals in combined population. Re-initializing population.")
            self.population = self._initialize_population()
            return self.population # Return the newly initialized (hopefully valid) population.

        # Sort the valid combined population by fitness in descending order (higher score is better).
        valid_individuals_for_next_gen.sort(
            key=lambda ind: self._calculate_fitness_for_individual(ind),
            reverse=True # Higher fitness scores are better
        )

        # Select the top individuals to form the next generation's population (elitism).
        self.population = valid_individuals_for_next_gen[:self.population_size]

        return self.population


# Print statement to confirm module loading.
# print(f"--- genetic_algorithm_core.py (version {__import__('datetime').date.today()}) loaded ---")