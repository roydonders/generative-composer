# ga_components/music_utils.py
"""
Module: music_utils.py

Purpose:
This module provides a collection of utility functions for music-related
operations, primarily leveraging the music21 library. These functions support
the genetic algorithm and other parts of the application by handling tasks such as:
- Parsing key signature strings into music21 Key objects.
- Generating lists of allowed pitches within a given key and octave range.
- Determining the tonal function of a pitch within a key (e.g., tonic, dominant).
- Analyzing the character of a musical phrase (e.g., "question", "answer", "neutral"),
  though this is less emphasized in the current fitness function.
- Segmenting a melody into phrases based on duration and extracting their properties,
  which is crucial for comparing evolved melodies against a reference structure.

Design Philosophy:
The functions here are designed to be general-purpose musical utilities. They encapsulate
music theory logic that can be complex, providing simpler interfaces for other modules.
Robustness and clear outputs are prioritized.
"""

# Third-party library imports (music21 for music theory objects and analysis)
from music21 import key, note, interval, pitch as m21pitch, roman  # roman is used for get_chord_tones

# Standard library imports
from typing import List, Tuple, Optional, Dict, Set, Any  # Set is used for chord tones

# Local application/library specific imports
# Import constants defining musical parameters, default values, etc.
from .music_constants import *


class MusicUtils:
    """
    A utility class containing static methods for music theory calculations
    and MIDI data analysis, primarily using the music21 library.
    These methods provide foundational musical knowledge and processing
    capabilities to other components of the application.
    """

    @staticmethod
    def get_key_object(key_name_str: str) -> key.Key:
        """
        Converts a key name string (e.g., "C major", "a minor", "f# dorian")
        into a music21.key.Key object. This method attempts several parsing
        strategies for robustness, as key strings can come in various formats.
        The input `key_name_str` is often derived from `str(music21.key.Key_object)`
        from an initial MIDI analysis, or user input.

        Args:
            key_name_str (str): The string representation of the key.

        Returns:
            music21.key.Key: A Key object representing the specified key.
                             Defaults to "C major" if all parsing attempts fail.
        """
        processed_key_str = key_name_str.strip()  # Remove leading/trailing whitespace.

        # Attempt 1: Direct parsing with the processed string.
        # music21's Key constructor is quite flexible.
        try:
            return key.Key(processed_key_str)
        except Exception as e1:
            # If direct parsing fails, try splitting into tonic and mode.
            try:
                parts = processed_key_str.split(' ', 1)  # Split only on the first space
                tonic_name = parts[0]
                mode_name = 'major'  # Default mode if not specified after tonic.

                if len(parts) > 1:
                    potential_mode = parts[1].strip()
                    # Normalize common mode names for music21's Key(tonic, mode) constructor.
                    if potential_mode.lower() == 'major':
                        mode_name = 'major'
                    elif potential_mode.lower() == 'minor':
                        mode_name = 'minor'
                    else:
                        # For other modes (Dorian, Phrygian, etc.) or unusual strings,
                        # pass the mode part as is. music21 might handle it.
                        mode_name = potential_mode

                return key.Key(tonic_name, mode_name)
            except Exception as e2:
                # Attempt 3: Try parsing just the first part as tonic, letting music21 infer the mode.
                # This is useful if key_name_str was "C" (becomes C major) or "a" (becomes a minor),
                # or if the mode part from the split was problematic.
                try:
                    # Use only the part before the first space as a potential tonic.
                    first_part_as_tonic = processed_key_str.split(' ')[0]
                    return key.Key(first_part_as_tonic)  # music21 infers major/minor for simple letter names.
                except Exception as e3:
                    # All attempts failed. Log the errors and default to C major.
                    print(
                        f"Warning (MusicUtils.get_key_object): Could not parse key '{key_name_str}' after multiple attempts. "
                        f"Errors: E1({type(e1).__name__}), E2({type(e2).__name__}), E3({type(e3).__name__}). Defaulting to C major.")
                    return key.Key("C", "major")  # Final, most robust fallback.

    @staticmethod
    def get_scale_pitches_for_key(key_name_str: str,
                                  octave_range: Tuple[int, int] = (MIN_OCTAVE, MAX_OCTAVE)
                                  ) -> List[str]:
        """
        Generates a list of all pitch names (e.g., "C4", "F#5") that belong to the
        specified key within a given octave range.

        Args:
            key_name_str (str): The string representation of the key.
            octave_range (Tuple[int, int]): A tuple (min_octave, max_octave) specifying
                                             the desired octave span. Defaults to values
                                             from `music_constants`.

        Returns:
            List[str]: A list of unique pitch strings (name with octave) in the key and range.
                       Returns a default C major scale if the specified key yields no pitches.
        """
        # Get the music21 Key object using the robust parsing method.
        scale_key_obj = MusicUtils.get_key_object(key_name_str)
        pitches_in_key: List[str] = []

        # Ensure octave range is valid and within reasonable music21 limits (0-8 typically).
        oct_start, oct_end = min(octave_range), max(octave_range)
        oct_start = max(0, min(8, oct_start))  # Clamp octaves to a sane range (0 to 8).
        oct_end = max(0, min(8, oct_end))
        if oct_start > oct_end:  # Ensure start is not greater than end after clamping.
            oct_start, oct_end = oct_end, oct_start  # Swap if necessary.

        # Iterate through the specified octave range.
        for oct_num in range(oct_start, oct_end + 1):
            # Get the abstract scale degrees (pitch classes without octave) for the current key.
            # music21's Key.getPitches() method is generally reliable for this.
            for p_abstract in scale_key_obj.getPitches():
                # Combine the pitch class name with the current octave number.
                p_with_octave = p_abstract.name + str(oct_num)
                pitches_in_key.append(p_with_octave)

        # Fallback: If, for some reason, no pitches were generated (e.g., problematic key object
        # or extremely narrow/invalid octave range not caught by clamping), default to C major.
        if not pitches_in_key:
            print(
                f"Warning (MusicUtils.get_scale_pitches_for_key): No pitches generated for key '{key_name_str}'. Defaulting to C major scale.")
            default_key_obj = key.Key("C", "major")  # Ensure mode is specified for default
            return [p.name + str(o) for o in range(MIN_OCTAVE, MAX_OCTAVE + 1)
                    for p in default_key_obj.getPitches()]

        return list(set(pitches_in_key))  # Return unique pitches to avoid duplicates if any.

    @staticmethod
    def get_tonal_function(pitch_obj: Optional[m21pitch.Pitch], current_key: key.Key) -> Optional[str]:
        """
        Determines the tonal function (e.g., "tonic", "dominant", "rest", "chromatic")
        of a given music21.pitch.Pitch object within the context of a specified key.

        Args:
            pitch_obj (Optional[music21.pitch.Pitch]): The Pitch object to analyze.
                                                       If None, it's treated as a rest.
            current_key (music21.key.Key): The Key object representing the current musical key.

        Returns:
            Optional[str]: A string describing the tonal function.
                           Returns "rest" if pitch_obj is None.
                           Returns "chromatic" if the pitch is out of the diatonic scale.
                           Returns "other_degree" if it's a diatonic degree not explicitly named.
                           Returns None on an unexpected error during analysis.
        """
        if pitch_obj is None:
            return "rest"  # Clearly a rest.

        try:
            # Get the scale degree of the pitch in the current key.
            # music21 scale degrees are 1-indexed (1=tonic, 2=supertonic, etc.).
            # `comparisonAttribute='name'` ensures comparison by pitch class name (ignoring octave for degree).
            degree = current_key.getScaleDegreeFromPitch(pitch_obj, comparisonAttribute='name')

            if degree is None:
                # If not directly in the diatonic scale, it's considered chromatic.
                # More advanced analysis could check for borrowed chords or specific alterations.
                return "chromatic"

            # Map scale degrees to common tonal function names.
            if degree == 1: return "tonic"
            if degree == 2: return "supertonic"
            if degree == 3: return "mediant"
            if degree == 4: return "subdominant"
            if degree == 5: return "dominant"
            if degree == 6: return "submediant"
            if degree == 7:
                # For the 7th degree, distinguish between leading tone (half-step below tonic)
                # and subtonic (whole-step below tonic), which have different functions.
                tonic_pitch = current_key.tonic
                # Calculate the interval from the current pitch to the tonic.
                # We are interested if the current pitch is *below* the tonic.
                interval_from_pitch_to_tonic = interval.Interval(pitch_obj, tonic_pitch)

                # A leading tone is a minor second below the tonic (m2 above it when inverted).
                # A subtonic is a major second below the tonic (M2 above it when inverted).
                # music21 interval names: 'm2' (minor second), 'M2' (major second).
                # Semitones: m2 = 1, M2 = 2.
                # If pitch_obj is the 7th degree, it's *below* the tonic (or an octave thereof).
                # So, interval from pitch_obj to tonic_pitch (an octave higher) would be m2 or M2.
                # Example: B to C (tonic) is m2. Bb to C (tonic) is M2.
                if abs(interval_from_pitch_to_tonic.semitones) == 1:  # or check interval_from_pitch_to_tonic.name == 'm2'
                    return "leading_tone"
                elif abs(
                        interval_from_pitch_to_tonic.semitones) == 2:  # or check interval_from_pitch_to_tonic.name == 'M2'
                    return "subtonic"
                # Fallback for other types of 7th degree (e.g., in modes without a strong leading tone tendency).
                return "degree7_ambiguous"  # e.g. natural minor's 7th.

            return "other_degree"  # Should not be reached if degrees are 1-7 and handled.

        except Exception as e:
            print(
                f"Error (MusicUtils.get_tonal_function): determining tonal function for pitch {pitch_obj} in key {current_key}: {e}")
            return None  # Error case.

    @staticmethod
    def analyze_phrase_character(phrase_notes: MelodySequence, current_key: key.Key) -> str:
        """
        Analyzes a musical phrase (sequence of notes/rests) to determine its
        likely character (e.g., "question", "answer", "neutral") based on its
        ending note's tonal function, duration, and melodic approach.
        This is a heuristic analysis and may not capture all nuances of musical phrasing.
        Its use in the current fitness function is limited to focus more on structural matching.

        Args:
            phrase_notes (MelodySequence): A list of (pitch, duration) tuples representing the phrase.
            current_key (music21.key.Key): The Key object for the current musical context.

        Returns:
            str: A string ("question", "answer", "neutral") indicating the phrase's character.
        """
        # Basic validation: phrase must exist and have a minimum number of notes.
        if not phrase_notes or not isinstance(phrase_notes, list) or \
                not all(isinstance(n, tuple) and len(n) == 2 for n in phrase_notes) or \
                len(phrase_notes) < MIN_NOTES_PER_PHRASE:  # Check against minimum notes for a phrase.
            return "neutral"  # Not enough information to determine character.

        # Get the last musical event (note or rest) in the phrase.
        last_event_pitch_str, last_event_duration = phrase_notes[-1]

        if last_event_pitch_str is None:  # Phrase ends with a rest.
            # A longer rest might imply resolution or a pause characteristic of an "answer" or conclusive end.
            return "answer" if last_event_duration >= 1.0 else "neutral"

        try:
            last_pitch_obj = m21pitch.Pitch(last_event_pitch_str)
            last_note_tonal_func = MusicUtils.get_tonal_function(last_pitch_obj, current_key)
        except Exception:
            return "neutral"  # Could not parse the last pitch or determine its function.

        if last_note_tonal_func is None:  # Tonal function could not be determined.
            return "neutral"

        # Consider melodic approach to the final note (semitones from penultimate to final note).
        melodic_approach_semitones = 0
        if len(phrase_notes) > 1:  # Need at least two events to determine approach.
            # Find the pitch of the event immediately preceding the final event.
            prev_event_pitch_str, _ = phrase_notes[-2]
            if prev_event_pitch_str is not None and last_pitch_obj:  # Only if previous was a note and current is a note.
                try:
                    prev_pitch_obj = m21pitch.Pitch(prev_event_pitch_str)
                    melodic_approach_semitones = interval.Interval(prev_pitch_obj, last_pitch_obj).semitones
                except Exception:
                    pass  # Ignore if previous pitch cannot be parsed or interval fails.

        # Heuristic rules for Question/Answer character based on ending note's function and context.
        # These are simplified and can be expanded for more nuanced analysis.
        # Constants QUESTION_ENDING_NOTES_STABILITY and ANSWER_ENDING_NOTES_STABILITY can be used here.
        if last_note_tonal_func in ANSWER_ENDING_NOTES_STABILITY:  # e.g., "tonic", "mediant"
            # Ending on tonic is often an "answer", especially if long or approached by step/downwards.
            if last_note_tonal_func == "tonic" and last_event_duration >= 1.0:  # Long tonic is quite conclusive.
                return "answer"
            # Descending or small interval approach to a stable tone can also feel like resolution.
            elif melodic_approach_semitones <= 0 and last_event_duration >= 0.5:
                return "answer"
            else:
                return "neutral"  # Stable tone but context is ambiguous.
        elif last_note_tonal_func in QUESTION_ENDING_NOTES_STABILITY:  # e.g., "dominant", "supertonic", "leading_tone"
            # Ending on dominant or leading tone often feels like a "question" (unresolved).
            # Shorter durations or upward approaches to these tones can enhance this feeling.
            if melodic_approach_semitones >= 0 and last_event_duration <= 1.0:
                return "question"
            elif last_event_duration < 1.0:  # Shorter notes on unstable degrees.
                return "question"
            else:
                return "neutral"  # Unstable tone but context is ambiguous.

        # Default to neutral if no strong characteristics are met.
        return "neutral"

    @staticmethod
    def get_tonal_hierarchy(current_key: key.Key) -> Dict[str, Optional[str]]:
        """
        Identifies key tonal centers (tonic, dominant, subdominant, mediant, leading_tone,
        supertonic, submediant) for a given key. These are useful for guiding melody
        generation towards stable points or for analysis.

        Args:
            current_key (music21.key.Key): The Key object.

        Returns:
            Dict[str, Optional[str]]: A dictionary mapping tonal function names
                                      (e.g., "tonic") to pitch class names (e.g., "C", "G#").
                                      Returns None for a function if it cannot be determined.
        """
        nodes: Dict[str, Optional[str]] = {
            "tonic": None, "dominant": None, "subdominant": None,
            "mediant": None, "leading_tone": None, "supertonic": None, "submediant": None
        }
        try:
            nodes["tonic"] = current_key.tonic.name
            # music21 uses 1-indexed degrees for pitchFromDegree.
            dom_pitch = current_key.pitchFromDegree(5)
            nodes["dominant"] = dom_pitch.name if dom_pitch else None

            sub_pitch = current_key.pitchFromDegree(4)
            nodes["subdominant"] = sub_pitch.name if sub_pitch else None

            med_pitch = current_key.pitchFromDegree(3)
            nodes["mediant"] = med_pitch.name if med_pitch else None

            # music21's getLeadingTone is more direct than pitchFromDegree(7) for this specific function.
            leading_tone_pitch = current_key.getLeadingTone()
            if leading_tone_pitch:
                nodes["leading_tone"] = leading_tone_pitch.name
            else:  # Fallback if getLeadingTone fails (e.g., for modes without a clear leading tone like natural minor's bVII).
                lt_fallback = current_key.pitchFromDegree(7)  # This would be the subtonic in some contexts.
                if lt_fallback: nodes[
                    "leading_tone"] = lt_fallback.name  # Store it as degree 7 if getLeadingTone is None

            sup_pitch = current_key.pitchFromDegree(2)
            nodes["supertonic"] = sup_pitch.name if sup_pitch else None

            submed_pitch = current_key.pitchFromDegree(6)
            nodes["submediant"] = submed_pitch.name if submed_pitch else None

        except Exception as e:
            # Fallback to C major hierarchy if an error occurs during analysis.
            print(
                f"Error (MusicUtils.get_tonal_hierarchy): getting tonal hierarchy for key {current_key}, defaulting for C major: {e}")
            c_maj_key = key.Key("C", "major")  # Ensure mode is specified for default
            nodes["tonic"] = c_maj_key.tonic.name
            nodes["dominant"] = c_maj_key.pitchFromDegree(5).name
            nodes["subdominant"] = c_maj_key.pitchFromDegree(4).name
            nodes["mediant"] = c_maj_key.pitchFromDegree(3).name
            nodes["leading_tone"] = c_maj_key.getLeadingTone().name  # C major has a clear leading tone (B)
            nodes["supertonic"] = c_maj_key.pitchFromDegree(2).name
            nodes["submediant"] = c_maj_key.pitchFromDegree(6).name
        return nodes

    @staticmethod
    def get_chord_tones(current_key: key.Key, scale_degree: int, num_triad_notes: int = 3) -> Set[str]:
        """
        Gets the pitch class names (e.g., "C", "E", "G") of the tones forming a triad
        built on a given scale degree within the specified key.

        Args:
            current_key (music21.key.Key): The Key object.
            scale_degree (int): The scale degree (1-7) on which the chord is built.
            num_triad_notes (int): Typically 3 for a triad (root, third, fifth).
                                   Can be adjusted for 7th chords, etc., though logic here is for triads.

        Returns:
            Set[str]: A set of pitch class names for the chord tones.
                      Returns an empty set if chord tones cannot be determined.
        """
        try:
            # RomanNumeral object in music21 represents a chord in a key context.
            # Degree should be an integer (1-7) or a Roman numeral string (e.g., "V", "ii").
            # This is the most robust way to get diatonic chord tones.
            roman_numeral_obj = roman.RomanNumeral(scale_degree, current_key)
            # .pitches provides the actual pitch objects of the chord. We want their names (pitch classes).
            return {p.name for p in roman_numeral_obj.pitches[:num_triad_notes]}
        except Exception as e_roman:
            # Fallback if RomanNumeral construction fails (e.g., invalid degree for the key's mode,
            # or if music21 has trouble with a specific modal context).
            # Attempt manual triad construction (simplified for major/minor common triads).
            try:
                root_pitch = current_key.pitchFromDegree(scale_degree)
                if not root_pitch: return set()  # Cannot build chord without a root.

                # Determine third type (major 'M3' or minor 'm3') based on key and degree for common triads.
                third_type_str = 'M3'  # Major third by default.
                if current_key.mode == 'minor':
                    # In minor (natural/harmonic): i, iv are minor; III+, V, VI, VII can be major (depending on form).
                    # For simplicity, assume i and iv are minor. V is often major (harmonic/melodic).
                    if scale_degree in [1, 4]: third_type_str = 'm3'  # i, iv are minor triads.
                    # ii° is diminished. III+ is augmented. For basic triads, these are more complex.
                elif current_key.mode == 'major':
                    # In major: I, IV, V are major; ii, iii, vi are minor; vii° is diminished.
                    if scale_degree in [2, 3, 6]:
                        third_type_str = 'm3'  # ii, iii, vi are minor triads.
                    elif scale_degree == 7:
                        third_type_str = 'm3'  # vii° has a minor third from root.

                p1 = root_pitch  # Root
                p2 = p1.transpose(third_type_str)  # Third

                # Determine fifth type (perfect 'P5' or diminished 'd5').
                fifth_type_str = 'P5'  # Perfect fifth by default.
                # Diminished triads occur on vii° in major and ii° in minor (natural/harmonic).
                if (current_key.mode == 'major' and scale_degree == 7) or \
                        (current_key.mode == 'minor' and scale_degree == 2):
                    fifth_type_str = 'd5'  # Diminished fifth.
                # Augmented triad (e.g., III+ in harmonic minor) would need 'A5'. Simplified here.

                p3 = p1.transpose(fifth_type_str)  # Fifth
                return {p.name for p in [p1, p2, p3] if p}  # Collect names of valid pitches.
            except Exception as e_manual:
                print(
                    f"Error (MusicUtils.get_chord_tones): getting chord tones for degree {scale_degree} in key {current_key}. RomanNumeral error: {e_roman}. Manual construction error: {e_manual}")
                return set()

    @staticmethod
    def analyze_reference_phrases(
            melody_events: MelodySequence,  # List of (pitch_string_or_None, duration_float)
            current_key: key.Key  # Expects a music21.key.Key object, not a string.
    ) -> List[ReferencePhraseInfo]:
        """
        Analyzes a reference melody (sequence of note/rest events) to segment it
        into musical phrases. Phrase segmentation is primarily based on cumulative
        duration, aiming for segments around `DEFAULT_PHRASE_DURATION_BEATS`.
        The character of each phrase (Question/Answer) is also heuristically analyzed.
        This information is crucial for the fitness function to compare evolved melodies
        against the reference's phrase structure and boundary notes.

        Args:
            melody_events: A list of (pitch_string_or_None, duration_float) tuples
                           representing the notes and rests of the reference melody.
            current_key: The music21.key.Key object for the melody's key, used for
                         tonal analysis within phrase characterization.

        Returns:
            List[ReferencePhraseInfo]: A list of ReferencePhraseInfo dictionaries,
                                       where each dictionary contains details about a
                                       segmented phrase. Returns an empty list if
                                       the input `melody_events` is empty.
        """
        if not melody_events:
            return []  # Cannot analyze an empty melody.

        phrases_info_list: List[ReferencePhraseInfo] = []
        current_phrase_note_events: MelodySequence = []  # Accumulates notes for the current phrase.
        current_phrase_total_duration_beats = 0.0  # Tracks duration of the current phrase.
        current_phrase_start_event_index = 0  # Start index in the original melody_events list.

        for i, (pitch_str, duration_val) in enumerate(melody_events):
            current_phrase_note_events.append((pitch_str, duration_val))
            current_phrase_total_duration_beats += duration_val

            # Condition to end the current phrase and finalize it:
            # 1. If the phrase duration meets or exceeds the default target phrase duration.
            # 2. OR, if this is the very last event in the entire melody sequence.
            is_last_event_in_melody = (i == len(melody_events) - 1)
            if current_phrase_total_duration_beats >= DEFAULT_PHRASE_DURATION_BEATS or is_last_event_in_melody:
                # Analyze the character (e.g., Question/Answer) of the completed phrase.
                # This uses tonal functions of ending notes, etc.
                phrase_char = MusicUtils.analyze_phrase_character(current_phrase_note_events, current_key)

                # Create the phrase information dictionary using the TypedDict structure.
                phrase_data: ReferencePhraseInfo = {
                    "start_index": current_phrase_start_event_index,
                    "end_index": i,  # Inclusive end index of the event in the original melody_events.
                    "num_notes": len(current_phrase_note_events),  # Number of events in this phrase.
                    "duration_beats": round(current_phrase_total_duration_beats, 3),  # Rounded for cleaner data.
                    "character": phrase_char,
                    "notes": list(current_phrase_note_events)  # Store a copy of the notes for this phrase.
                }
                phrases_info_list.append(phrase_data)

                # Reset accumulators for the next phrase.
                current_phrase_note_events = []
                current_phrase_total_duration_beats = 0.0
                current_phrase_start_event_index = i + 1  # Next phrase starts after the current event.

        return phrases_info_list

# print(f"--- music_utils.py (Comprehensive Comments, v. {__import__('datetime').date.today()}) loaded ---")
