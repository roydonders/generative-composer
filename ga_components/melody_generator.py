# ga_components/melody_generator.py
"""
Module: melody_generator.py

Purpose:
This module is responsible for the procedural generation and modification
(mutation) of musical melodies within the genetic algorithm. It uses musical
rules and probabilistic choices to create initial melodies and to introduce
variations during the evolutionary process.

Key functionalities include:
- Initialization with a specific musical key.
- Generation of new melodies based on target length, phrase structures, and musical heuristics.
- Mutation of existing melodies by altering pitches, rhythms, or adding/removing notes/rests.
- Application of motif development techniques for musical coherence.
"""

import random
from typing import List, Tuple, Optional, Dict, Set # Set, Dict unused here

# Import music21 components
from music21 import note, interval, pitch as m21pitch, key # Added key for type hint

# Import necessary constants and type definitions
from .music_constants import (
    MIN_OCTAVE, MAX_OCTAVE, POSSIBLE_DURATIONS,
    MOTIF_REPETITION_CHANCE, INTERNAL_PHRASE_REPETITION_CHANCE,
    PHRASE_PUNCTUATION_CHANCE, REST_PROBABILITY_GENERATION,
    MAX_CONSECUTIVE_REPEATED_PITCHES, MelodySequence, MelodyNote,
    NOTES_PER_PHRASE_FALLBACK, DEFAULT_PHRASE_DURATION_BEATS,
    MUTATION_RATE,
    REST_PROBABILITY_MUTATION, PHRASE_END_MUTATION_CHANCE, MIN_NOTES_PER_PHRASE
)
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
        Initializes the MelodyGenerator.

        Args:
            key_name_str (str): The musical key for generation (e.g., "C major", "g minor").
            notes_per_phrase_fallback (int): Default number of notes per phrase if not
                                             guided by reference durations.
            target_phrase_duration_beats (float): Default target duration for generated phrases in beats.
        """
        self.key_name: str = key_name_str
        # Corrected type hint from m21pitch.Key to key.Key
        self.key_obj: key.Key = MusicUtils.get_key_object(key_name_str)

        self.allowed_pitches: List[str] = MusicUtils.get_scale_pitches_for_key(
            key_name_str, (MIN_OCTAVE, MAX_OCTAVE)
        )
        self.tonal_nodes: Dict[str, Optional[str]] = MusicUtils.get_tonal_hierarchy(self.key_obj)
        self.common_chord_tones_sets: List[Set[str]] = [
            MusicUtils.get_chord_tones(self.key_obj, degree) for degree in [1, 2, 3, 4, 5, 6]
            if MusicUtils.get_chord_tones(self.key_obj, degree)
        ]
        self.notes_per_phrase_fallback: int = notes_per_phrase_fallback
        self.target_phrase_duration_beats: float = target_phrase_duration_beats
        self.weighted_pitches: List[str] = self._create_weighted_pitches()

        if not self.allowed_pitches:
            print(f"Warning: No allowed pitches found for key {key_name_str}. Defaulting to C major pitches.")
            self.allowed_pitches = [p + str(o) for o in range(MIN_OCTAVE, MAX_OCTAVE + 1) for p in "CDEFGAB"]
        if not self.weighted_pitches:
            self.weighted_pitches = list(self.allowed_pitches)

    def _create_weighted_pitches(self) -> List[str]:
        """
        Creates a list of allowed pitches where tonally important notes (tonic, dominant, etc.)
        are repeated, effectively increasing their probability of being chosen.
        (Implementation unchanged)
        """
        weighted_list: List[str] = []
        if not self.allowed_pitches:
            return ['C4']

        for pitch_name_octave in self.allowed_pitches:
            try:
                current_pitch_obj = m21pitch.Pitch(pitch_name_octave)
                pitch_class_name = current_pitch_obj.name
                weight = 1
                if self.tonal_nodes.get("tonic") and pitch_class_name == self.tonal_nodes["tonic"]:
                    weight = 8
                elif self.tonal_nodes.get("dominant") and pitch_class_name == self.tonal_nodes["dominant"]:
                    weight = 6
                elif self.tonal_nodes.get("subdominant") and pitch_class_name == self.tonal_nodes["subdominant"]:
                    weight = 4
                elif self.tonal_nodes.get("mediant") and pitch_class_name == self.tonal_nodes["mediant"]:
                    weight = 3
                weighted_list.extend([pitch_name_octave] * weight)
            except Exception as e:
                print(f"Warning: Could not parse pitch {pitch_name_octave} for weighting: {e}")
                weighted_list.append(pitch_name_octave)
        return weighted_list if weighted_list else list(self.allowed_pitches)

    def _is_justified_leap(self, prev_pitch_obj: Optional[m21pitch.Pitch],
                           current_pitch_obj: Optional[m21pitch.Pitch]) -> bool:
        """
        Checks if a melodic leap between two pitches can be considered "justified".
        (Implementation unchanged)
        """
        if not prev_pitch_obj or not current_pitch_obj:
            return False
        for chord_tone_set in self.common_chord_tones_sets:
            if prev_pitch_obj.name in chord_tone_set and current_pitch_obj.name in chord_tone_set:
                return True
        return False

    def _get_next_note_tuple(self,
                             previous_pitch_obj: Optional[m21pitch.Pitch],
                             last_leap_semitones: int,
                             time_left_in_phrase: float,
                             is_phrase_end_approaching: bool,
                             current_phrase_duration_beats: float,
                             target_duration_this_phrase: float
                            ) -> MelodyNote:
        """
        Selects the next pitch and duration for a melody.
        (Implementation unchanged)
        """
        chosen_pitch_str: Optional[str]
        if random.random() < REST_PROBABILITY_GENERATION and not is_phrase_end_approaching:
            chosen_pitch_str = None
        else:
            if previous_pitch_obj is None or not self.allowed_pitches:
                chosen_pitch_str = random.choice(
                    self.weighted_pitches if self.weighted_pitches else ['C4']
                )
            else:
                candidate_pitches_weighted: List[Tuple[str, int]] = []
                for p_str_candidate in self.allowed_pitches:
                    try:
                        candidate_obj = m21pitch.Pitch(p_str_candidate)
                        if abs(candidate_obj.octave - previous_pitch_obj.octave) > 1 and \
                           abs(candidate_obj.ps - previous_pitch_obj.ps) > 12:
                            continue
                        interval_val = interval.Interval(previous_pitch_obj, candidate_obj).semitones
                        abs_interval = abs(interval_val)
                        weight = 0
                        if abs_interval == 0: weight = 1
                        elif abs_interval <= 2: weight = 30
                        elif abs_interval <= 4: weight = 10
                        elif abs_interval <= 7:
                            if self._is_justified_leap(previous_pitch_obj, candidate_obj): weight = 5
                            else: weight = 2
                        else: weight = 1
                        if abs(last_leap_semitones) > 7:
                            if abs_interval <= 2 and (interval_val * last_leap_semitones < 0): weight *= 3
                        if weight > 0:
                            candidate_pitches_weighted.append((p_str_candidate, int(weight)))
                    except Exception:
                        continue
                if not candidate_pitches_weighted:
                    chosen_pitch_str = random.choice(
                        self.weighted_pitches if self.weighted_pitches else ['C4']
                    )
                else:
                    choices, weights = zip(*candidate_pitches_weighted)
                    chosen_pitch_str = random.choices(choices, weights=weights, k=1)[0]

        possible_rhythms = [r for r in POSSIBLE_DURATIONS if r <= time_left_in_phrase + 0.25]
        chosen_duration: float
        if not possible_rhythms:
            chosen_duration = min(POSSIBLE_DURATIONS) if time_left_in_phrase > 0.1 else max(0.125, time_left_in_phrase)
        elif is_phrase_end_approaching:
            fitting_rhythms = [r for r in possible_rhythms if abs(r - time_left_in_phrase) < 0.125]
            if fitting_rhythms and random.random() < 0.7: chosen_duration = random.choice(fitting_rhythms)
            else: chosen_duration = random.choice(possible_rhythms)
        elif current_phrase_duration_beats < target_duration_this_phrase * 0.4:
            chosen_duration = random.choice([d for d in possible_rhythms if d <= 1.0] or possible_rhythms)
        else:
            chosen_duration = random.choice(possible_rhythms)
        chosen_duration = max(0.25, chosen_duration)
        if chosen_pitch_str is None and not is_phrase_end_approaching:
            chosen_duration = random.choice([d for d in POSSIBLE_DURATIONS if d <= 1.0])
        return (chosen_pitch_str, chosen_duration)

    def _apply_motif_development(self, motif_to_develop: MelodySequence) -> MelodySequence:
        """
        Applies simple transformations to a given melodic motif.
        (Implementation unchanged)
        """
        if not motif_to_develop: return []
        developed_motif = list(motif_to_develop)
        rand_variation = random.random()
        if rand_variation < 0.6:
            interval_choices = [-4, -3, -2, 2, 3, 4]
            semitones_to_transpose = random.choice(interval_choices)
            temp_developed_motif: MelodySequence = []
            valid_development = True
            for pitch_str, duration in developed_motif:
                if pitch_str is None:
                    temp_developed_motif.append((None, duration))
                    continue
                try:
                    original_pitch = m21pitch.Pitch(pitch_str)
                    transposed_pitch_obj = original_pitch.transpose(semitones_to_transpose)
                    if MIN_OCTAVE <= transposed_pitch_obj.octave <= MAX_OCTAVE and \
                       any(allowed_p.startswith(transposed_pitch_obj.name) for allowed_p in self.allowed_pitches):
                        temp_developed_motif.append((transposed_pitch_obj.nameWithOctave, duration))
                    else:
                        valid_development = False; break
                except Exception:
                    valid_development = False; break
            if valid_development: return temp_developed_motif
        elif rand_variation < 0.9:
            for k_motif in range(len(developed_motif)):
                pitch_str, duration = developed_motif[k_motif]
                if pitch_str is not None and random.random() < 0.7:
                    new_duration_options = [d for d in POSSIBLE_DURATIONS if abs(d - duration) > 0.125]
                    if new_duration_options:
                        new_duration = random.choice(new_duration_options)
                        developed_motif[k_motif] = (pitch_str, new_duration)
            return developed_motif
        return developed_motif

    def generate_melody(self, total_notes_target: int,
                        ref_phrase_durations: Optional[List[float]] = None) -> MelodySequence:
        """
        Generates a complete melody composed of several phrases.
        (Implementation unchanged)
        """
        melody: MelodySequence = []
        generated_phrases: List[MelodySequence] = []
        num_phrases_to_generate = (len(ref_phrase_durations) if ref_phrase_durations
                                   else max(1, total_notes_target // NOTES_PER_PHRASE_FALLBACK))
        num_phrases_to_generate = max(1, num_phrases_to_generate)
        previous_pitch_obj: Optional[m21pitch.Pitch] = None
        consecutive_repeated_count = 0
        last_leap_semitones = 0

        for phrase_idx in range(num_phrases_to_generate):
            current_phrase_notes: MelodySequence = []
            current_phrase_duration_beats = 0.0
            target_duration_this_phrase = (ref_phrase_durations[phrase_idx]
                                           if ref_phrase_durations and phrase_idx < len(ref_phrase_durations)
                                           else self.target_phrase_duration_beats)
            if generated_phrases and random.random() < INTERNAL_PHRASE_REPETITION_CHANCE:
                phrase_to_develop_options = [
                    p for p in generated_phrases
                    if sum(n[1] for n in p) < target_duration_this_phrase * 1.5
                ]
                if phrase_to_develop_options:
                    phrase_to_develop = random.choice(phrase_to_develop_options)
                    developed_phrase = self._apply_motif_development(phrase_to_develop)
                    developed_phrase_dur = sum(n[1] for n in developed_phrase)
                    if developed_phrase and developed_phrase_dur <= target_duration_this_phrase * 1.2:
                        current_phrase_notes.extend(developed_phrase)
                        current_phrase_duration_beats += developed_phrase_dur
                        if developed_phrase[-1][0] is not None:
                            try: previous_pitch_obj = m21pitch.Pitch(developed_phrase[-1][0])
                            except: pass

            max_notes_this_phrase = int(target_duration_this_phrase / 0.25) if target_duration_this_phrase > 0 else NOTES_PER_PHRASE_FALLBACK * 2
            max_notes_this_phrase = max(MIN_NOTES_PER_PHRASE, min(max_notes_this_phrase, NOTES_PER_PHRASE_FALLBACK * 3))
            notes_generated_this_phrase_count = 0

            while (current_phrase_duration_beats < target_duration_this_phrase * 0.90 and
                   notes_generated_this_phrase_count < max_notes_this_phrase):
                time_left_for_phrase = target_duration_this_phrase - current_phrase_duration_beats
                is_ending_phrase_soon = time_left_for_phrase < 1.0
                next_pitch_str, next_duration = self._get_next_note_tuple(
                    previous_pitch_obj, last_leap_semitones, time_left_for_phrase, is_ending_phrase_soon,
                    current_phrase_duration_beats, target_duration_this_phrase
                )
                if next_pitch_str is not None:
                    if previous_pitch_obj and next_pitch_str == previous_pitch_obj.nameWithOctave:
                        consecutive_repeated_count += 1
                        if consecutive_repeated_count >= MAX_CONSECUTIVE_REPEATED_PITCHES and random.random() < 0.95:
                            forced_next_pitch_str, _ = self._get_next_note_tuple(
                                None, last_leap_semitones, time_left_for_phrase, is_ending_phrase_soon,
                                current_phrase_duration_beats, target_duration_this_phrase
                            )
                            if forced_next_pitch_str != next_pitch_str:
                                next_pitch_str = forced_next_pitch_str
                                consecutive_repeated_count = 0
                    else: consecutive_repeated_count = 0
                else: consecutive_repeated_count = 0
                current_phrase_notes.append((next_pitch_str, next_duration))
                current_phrase_duration_beats += next_duration
                notes_generated_this_phrase_count += 1
                if next_pitch_str is not None:
                    try:
                        current_pitch_obj = m21pitch.Pitch(next_pitch_str)
                        if previous_pitch_obj: last_leap_semitones = interval.Interval(previous_pitch_obj, current_pitch_obj).semitones
                        else: last_leap_semitones = 0
                        previous_pitch_obj = current_pitch_obj
                    except Exception:
                        last_leap_semitones = 0
                else: last_leap_semitones = 0
            if current_phrase_notes and random.random() < PHRASE_PUNCTUATION_CHANCE:
                last_note_idx = len(current_phrase_notes) - 1
                last_item_pitch, last_item_duration = current_phrase_notes[last_note_idx]
                if last_item_pitch is not None and last_item_duration < 1.0:
                    if current_phrase_duration_beats >= target_duration_this_phrase * 0.70:
                        new_ending_duration = random.choice([1.0, 1.5, 2.0])
                        current_phrase_duration_beats = current_phrase_duration_beats - last_item_duration + new_ending_duration
                        current_phrase_notes[last_note_idx] = (last_item_pitch, new_ending_duration)
            if current_phrase_notes:
                melody.extend(current_phrase_notes)
                generated_phrases.append(list(current_phrase_notes))
                if current_phrase_notes[-1][0] is not None:
                    try: previous_pitch_obj = m21pitch.Pitch(current_phrase_notes[-1][0])
                    except: pass
            else: break
        return melody

    def mutate_melody(self, melody: MelodySequence, mutation_rate_param: float = MUTATION_RATE) -> MelodySequence:
        """
        Applies mutations to an existing melody.
        (Implementation unchanged)
        """
        if not melody: return []
        mutated_melody = list(melody)
        previous_pitch_obj_for_mutation: Optional[m21pitch.Pitch] = None
        last_leap_for_mutation = 0

        for i in range(len(mutated_melody)):
            original_pitch_str, original_duration = mutated_melody[i]
            new_pitch_str, new_duration = original_pitch_str, float(original_duration)
            if random.random() < mutation_rate_param:
                if new_pitch_str is None:
                    if random.random() > REST_PROBABILITY_MUTATION:
                        new_pitch_str = self._get_next_pitch_for_mutation(previous_pitch_obj_for_mutation, last_leap_for_mutation)
                else:
                    if random.random() < REST_PROBABILITY_MUTATION: new_pitch_str = None
                    else: new_pitch_str = self._get_next_pitch_for_mutation(previous_pitch_obj_for_mutation, last_leap_for_mutation, current_pitch_to_avoid=new_pitch_str)
            if random.random() < mutation_rate_param:
                new_duration = random.choice(POSSIBLE_DURATIONS)
                if new_pitch_str is None and new_duration > 1.0:
                    new_duration = random.choice([d for d in POSSIBLE_DURATIONS if d <= 1.0])
            mutated_melody[i] = (new_pitch_str, new_duration)
            is_approx_phrase_end = (i + 1) % NOTES_PER_PHRASE_FALLBACK == 0 and i > 0
            if is_approx_phrase_end and random.random() < PHRASE_END_MUTATION_CHANCE:
                current_p, current_d = mutated_melody[i]
                if current_p is not None and current_d < 1.0:
                    mutated_melody[i] = (current_p, random.choice([d for d in POSSIBLE_DURATIONS if d >= 1.0] or [1.0]))
            try:
                if new_pitch_str is not None:
                    newly_set_pitch_obj = m21pitch.Pitch(new_pitch_str)
                    if previous_pitch_obj_for_mutation:
                        last_leap_for_mutation = interval.Interval(previous_pitch_obj_for_mutation, newly_set_pitch_obj).semitones
                    else: last_leap_for_mutation = 0
                    previous_pitch_obj_for_mutation = newly_set_pitch_obj
                else:
                    last_leap_for_mutation = 0
            except Exception:
                last_leap_for_mutation = 0
                previous_pitch_obj_for_mutation = None
        return mutated_melody

    def _get_next_pitch_for_mutation(self,
                                     previous_pitch_obj: Optional[m21pitch.Pitch],
                                     last_leap_semitones: int,
                                     current_pitch_to_avoid: Optional[str] = None) -> str:
        """
        Helper function to select a new pitch during mutation.
        (Implementation unchanged)
        """
        if not self.allowed_pitches: return 'C4'
        if previous_pitch_obj is None:
            candidate_pitches = [p for p in self.weighted_pitches if p != current_pitch_to_avoid]
            return random.choice(candidate_pitches if candidate_pitches else self.weighted_pitches)
        candidate_pitches_weighted: List[Tuple[str, int]] = []
        for p_str_candidate in self.allowed_pitches:
            if p_str_candidate == current_pitch_to_avoid: continue
            try:
                candidate_obj = m21pitch.Pitch(p_str_candidate)
                if abs(candidate_obj.octave - previous_pitch_obj.octave) > 1 and \
                   abs(candidate_obj.ps - previous_pitch_obj.ps) > 12: continue
                interval_val = interval.Interval(previous_pitch_obj, candidate_obj).semitones
                abs_interval = abs(interval_val)
                weight = 0
                if abs_interval <= 2: weight = 15
                elif abs_interval <= 4: weight = 5
                elif abs_interval <= 7: weight = 2
                else: weight = 1
                if abs(last_leap_semitones) > 7:
                    if abs_interval <= 2 and (interval_val * last_leap_semitones < 0): weight *= 2
                if weight > 0: candidate_pitches_weighted.append((p_str_candidate, weight))
            except Exception: continue
        if not candidate_pitches_weighted:
            fallback_options = [p for p in self.allowed_pitches if p != current_pitch_to_avoid]
            return random.choice(fallback_options if fallback_options else self.allowed_pitches)
        choices, weights = zip(*candidate_pitches_weighted)
        return random.choices(choices, weights=weights, k=1)[0]

# print(f"--- melody_generator.py (version {__import__('datetime').date.today()}) loaded ---")
