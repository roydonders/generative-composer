# ga_components/music_utils.py
"""
Module: music_utils.py

Purpose:
This module provides a collection of utility functions for music-related
operations. These functions support the genetic algorithm by handling tasks
such as key signature parsing, scale generation, tonal analysis, and
phrase characterization. They often leverage the music21 library for
complex music theory calculations.
"""

from music21 import key, note, interval, pitch as m21pitch, roman  # stream, roman were unused here

from typing import List, Tuple, Optional, Dict, Set, Any

from .music_constants import (
    MIN_OCTAVE, MAX_OCTAVE, MelodySequence, ReferencePhraseInfo,
    DEFAULT_PHRASE_DURATION_BEATS, TONAL_FUNCTION_MAP,
    QUESTION_ENDING_NOTES_STABILITY, ANSWER_ENDING_NOTES_STABILITY,
    MIN_NOTES_PER_PHRASE  # Added for use in analyze_phrase_character or similar logic
)


class MusicUtils:
    """
    A utility class containing static methods for music theory calculations
    and MIDI data analysis, primarily using the music21 library.
    """

    @staticmethod
    def get_key_object(key_name_str: str) -> key.Key:
        """
        Converts a key name string (e.g., "C major", "a minor") into a music21 Key object.
        Provides fallbacks and more robust parsing attempts.
        The input `key_name_str` is expected to be potentially from `str(music21.key.Key_object)`.

        Args:
            key_name_str: The string representation of the key.

        Returns:
            A music21.key.Key object representing the specified key. Defaults to "C major".
        """
        processed_key_str = key_name_str.strip()

        # Attempt 1: Direct parsing with the processed string
        try:
            # print(f"DEBUG: MusicUtils.get_key_object trying (Attempt 1): key.Key('{processed_key_str}')")
            return key.Key(processed_key_str)
        except Exception as e1:
            # print(f"DEBUG: MusicUtils.get_key_object attempt 1 failed for '{processed_key_str}': {type(e1).__name__}: {e1}")

            # Attempt 2: Split into tonic and mode, try to normalize mode
            try:
                parts = processed_key_str.split(' ', 1)
                tonic_name = parts[0]
                mode_name = 'major'  # Default if no mode part

                if len(parts) > 1:
                    potential_mode = parts[1]
                    # Normalize common mode names to lowercase for music21's Key(tonic, mode) constructor
                    # music21's Key(tonic, mode) is generally good with modes like "Major", "Minor", "Dorian" etc.
                    # but explicit lowercasing for "major"/"minor" can be safer if they appear capitalized.
                    if potential_mode.lower() == 'major':
                        mode_name = 'major'
                    elif potential_mode.lower() == 'minor':
                        mode_name = 'minor'
                    else:
                        # For other modes (Dorian, Phrygian, etc.) or unusual strings,
                        # pass the mode part as is. music21 might handle it.
                        mode_name = potential_mode

                # print(f"DEBUG: MusicUtils.get_key_object trying (Attempt 2): key.Key('{tonic_name}', '{mode_name}')")
                return key.Key(tonic_name, mode_name)
            except Exception as e2:
                # print(f"DEBUG: MusicUtils.get_key_object attempt 2 failed for tonic='{tonic_name}', mode='{mode_name}': {type(e2).__name__}: {e2}")

                # Attempt 3: Try parsing just the first part as tonic, let music21 infer mode
                # This is useful if key_name_str was "C" (becomes C major) or "a" (becomes a minor)
                # or if the mode part from split was problematic.
                try:
                    first_part_as_tonic = processed_key_str.split(' ')[0]
                    # print(f"DEBUG: MusicUtils.get_key_object trying (Attempt 3): key.Key('{first_part_as_tonic}') with mode inference")
                    return key.Key(first_part_as_tonic)
                except Exception as e3:
                    # print(f"DEBUG: MusicUtils.get_key_object attempt 3 failed for '{first_part_as_tonic}': {type(e3).__name__}: {e3}")
                    print(
                        f"Warning: Could not parse key '{key_name_str}' after multiple attempts. Errors: E1({type(e1).__name__}), E2({type(e2).__name__}), E3({type(e3).__name__}). Defaulting to C major.")
                    return key.Key("C", "major")  # Final, most robust fallback

    @staticmethod
    def get_scale_pitches_for_key(key_name_str: str,
                                  octave_range: Tuple[int, int] = (MIN_OCTAVE, MAX_OCTAVE)
                                  ) -> List[str]:
        """
        Generates a list of all pitch names that belong to the specified key
        within a given octave range. (Implementation mostly unchanged)
        """
        scale_key_obj = MusicUtils.get_key_object(key_name_str)  # Uses the robust getter
        pitches_in_key: List[str] = []
        oct_start, oct_end = min(octave_range), max(octave_range)
        oct_start = max(0, min(8, oct_start))
        oct_end = max(0, min(8, oct_end))
        if oct_start > oct_end: oct_start, oct_end = oct_end, oct_start

        for oct_num in range(oct_start, oct_end + 1):
            for p_abstract in scale_key_obj.getPitches():
                p_with_octave = p_abstract.name + str(oct_num)
                pitches_in_key.append(p_with_octave)

        if not pitches_in_key:
            print(f"Warning: No pitches generated for key '{key_name_str}'. Defaulting to C major scale.")
            default_key_obj = key.Key("C", "major")
            return [p.name + str(o) for o in range(MIN_OCTAVE, MAX_OCTAVE + 1)
                    for p in default_key_obj.getPitches()]
        return list(set(pitches_in_key))

    @staticmethod
    def get_tonal_function(pitch_obj: Optional[m21pitch.Pitch], current_key: key.Key) -> Optional[str]:
        """
        Determines the tonal function of a given music21 Pitch object.
        (Implementation mostly unchanged)
        """
        if pitch_obj is None: return "rest"
        try:
            degree = current_key.getScaleDegreeFromPitch(pitch_obj, comparisonAttribute='name')
            if degree is None: return "chromatic"
            if degree == 1: return "tonic"
            if degree == 2: return "supertonic"
            if degree == 3: return "mediant"
            if degree == 4: return "subdominant"
            if degree == 5: return "dominant"
            if degree == 6: return "submediant"
            if degree == 7:
                tonic_pitch = current_key.tonic
                interval_to_tonic = interval.Interval(pitch_obj, tonic_pitch)
                # Check semitones for leading tone (1 semitone below tonic) vs subtonic (2 semitones below)
                if interval_to_tonic.semitones == 1 or interval_to_tonic.semitones == -11:  # m2 above, or M7 below (enharmonically leading tone)
                    return "leading_tone"
                elif interval_to_tonic.semitones == 2 or interval_to_tonic.semitones == -10:  # M2 above, or m7 below (enharmonically subtonic)
                    return "subtonic"
                return "degree7_ambiguous"
            return "other_degree"
        except Exception as e:
            print(f"Error determining tonal function for pitch {pitch_obj} in key {current_key}: {e}")
            return None

    @staticmethod
    def analyze_phrase_character(phrase_notes: MelodySequence, current_key: key.Key) -> str:
        """
        Analyzes a musical phrase to determine its likely character (question, answer, neutral).
        (Implementation mostly unchanged)
        """
        if not phrase_notes or not isinstance(phrase_notes, list) or \
                not all(isinstance(n, tuple) and len(n) == 2 for n in phrase_notes) or \
                len(phrase_notes) < MIN_NOTES_PER_PHRASE:  # Added min notes check
            return "neutral"

        last_event_pitch_str, last_event_duration = phrase_notes[-1]
        if last_event_pitch_str is None:
            return "answer" if last_event_duration >= 1.0 else "neutral"

        try:
            last_pitch_obj = m21pitch.Pitch(last_event_pitch_str)
            last_note_func = MusicUtils.get_tonal_function(last_pitch_obj, current_key)
        except Exception:
            return "neutral"
        if last_note_func is None: return "neutral"

        melodic_approach_semitones = 0
        if len(phrase_notes) > 1:
            prev_event_pitch_str, _ = phrase_notes[-2]
            if prev_event_pitch_str is not None:
                try:
                    prev_pitch_obj = m21pitch.Pitch(prev_event_pitch_str)
                    melodic_approach_semitones = interval.Interval(prev_pitch_obj, last_pitch_obj).semitones
                except Exception:
                    pass

        if last_note_func == "tonic":
            if last_event_duration >= 1.0:
                return "answer"
            elif melodic_approach_semitones <= 0 and last_event_duration >= 0.5:
                return "answer"
            else:
                return "neutral"
        elif last_note_func == "dominant":
            if melodic_approach_semitones >= 0 and last_event_duration <= 1.0:
                return "question"
            elif last_event_duration < 1.0:
                return "question"
            else:
                return "neutral"
        elif last_note_func == "leading_tone":
            return "question"
        elif last_note_func == "mediant" and melodic_approach_semitones > 0 and last_event_duration <= 1.0:
            return "question"
        return "neutral"

    @staticmethod
    def get_tonal_hierarchy(current_key: key.Key) -> Dict[str, Optional[str]]:
        """
        Identifies key tonal centers for a given key. (Implementation mostly unchanged)
        """
        nodes: Dict[str, Optional[str]] = {
            "tonic": None, "dominant": None, "subdominant": None,
            "mediant": None, "leading_tone": None, "supertonic": None, "submediant": None
        }
        try:
            nodes["tonic"] = current_key.tonic.name
            dom_pitch = current_key.pitchFromDegree(5);
            nodes["dominant"] = dom_pitch.name if dom_pitch else None
            sub_pitch = current_key.pitchFromDegree(4);
            nodes["subdominant"] = sub_pitch.name if sub_pitch else None
            med_pitch = current_key.pitchFromDegree(3);
            nodes["mediant"] = med_pitch.name if med_pitch else None
            leading_tone_pitch = current_key.getLeadingTone()
            if leading_tone_pitch:
                nodes["leading_tone"] = leading_tone_pitch.name
            else:
                lt_fallback = current_key.pitchFromDegree(7)
                if lt_fallback: nodes["leading_tone"] = lt_fallback.name
            sup_pitch = current_key.pitchFromDegree(2);
            nodes["supertonic"] = sup_pitch.name if sup_pitch else None
            submed_pitch = current_key.pitchFromDegree(6);
            nodes["submediant"] = submed_pitch.name if submed_pitch else None
        except Exception as e:
            print(f"Error getting tonal hierarchy for key {current_key}, defaulting for C major: {e}")
            c_maj_key = key.Key("C", "major")  # Ensure mode is specified
            nodes["tonic"] = c_maj_key.tonic.name
            nodes["dominant"] = c_maj_key.pitchFromDegree(5).name
            nodes["subdominant"] = c_maj_key.pitchFromDegree(4).name
            nodes["mediant"] = c_maj_key.pitchFromDegree(3).name
            nodes["leading_tone"] = c_maj_key.getLeadingTone().name
            nodes["supertonic"] = c_maj_key.pitchFromDegree(2).name
            nodes["submediant"] = c_maj_key.pitchFromDegree(6).name
        return nodes

    @staticmethod
    def get_chord_tones(current_key: key.Key, scale_degree: int, num_triad_notes: int = 3) -> Set[str]:
        """
        Gets the pitch class names of the tones forming a triad built on a given scale degree.
        (Implementation mostly unchanged)
        """
        try:
            roman_numeral = roman.RomanNumeral(scale_degree, current_key)
            return {p.name for p in roman_numeral.pitches[:num_triad_notes]}
        except Exception as e_roman:
            try:
                root_pitch = current_key.pitchFromDegree(scale_degree)
                if not root_pitch: return set()
                third_type_str = 'M3'
                if current_key.mode == 'minor':
                    if scale_degree in [1, 4]: third_type_str = 'm3'
                elif current_key.mode == 'major':
                    if scale_degree in [2, 3, 6]:
                        third_type_str = 'm3'
                    elif scale_degree == 7:
                        third_type_str = 'm3'
                p1 = root_pitch;
                p2 = p1.transpose(third_type_str)
                fifth_type_str = 'P5'
                if (current_key.mode == 'major' and scale_degree == 7) or \
                        (current_key.mode == 'minor' and scale_degree == 2):
                    fifth_type_str = 'd5'
                p3 = p1.transpose(fifth_type_str)
                return {p.name for p in [p1, p2, p3] if p}
            except Exception as e_manual:
                print(
                    f"Error getting chord tones for degree {scale_degree} in key {current_key}: {e_roman} then {e_manual}")
                return set()

    @staticmethod
    def analyze_reference_phrases(
            melody_events: MelodySequence,
            current_key: key.Key  # Expects a Key object, not a string
    ) -> List[ReferencePhraseInfo]:
        """
        Analyzes a reference melody to segment it into musical phrases.
        (Implementation mostly unchanged)
        """
        if not melody_events: return []
        phrases_info_list: List[ReferencePhraseInfo] = []
        current_phrase_note_events: MelodySequence = []
        current_phrase_total_duration = 0.0
        current_phrase_start_index = 0

        for i, (pitch_str, duration_val) in enumerate(melody_events):
            current_phrase_note_events.append((pitch_str, duration_val))
            current_phrase_total_duration += duration_val
            is_last_event_in_melody = (i == len(melody_events) - 1)
            if current_phrase_total_duration >= DEFAULT_PHRASE_DURATION_BEATS or is_last_event_in_melody:
                phrase_char = MusicUtils.analyze_phrase_character(current_phrase_note_events, current_key)
                phrase_data: ReferencePhraseInfo = {
                    "start_index": current_phrase_start_index, "end_index": i,
                    "num_notes": len(current_phrase_note_events),
                    "duration_beats": round(current_phrase_total_duration, 3),
                    "character": phrase_char, "notes": list(current_phrase_note_events)
                }
                phrases_info_list.append(phrase_data)
                current_phrase_note_events = []
                current_phrase_total_duration = 0.0
                current_phrase_start_index = i + 1
        return phrases_info_list

# print(f"--- music_utils.py (version {__import__('datetime').date.today()}) loaded ---")
