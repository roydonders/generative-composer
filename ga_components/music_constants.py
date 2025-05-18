# ga_components/music_constants.py
"""
Module: music_constants.py

Purpose:
This module serves as a central repository for all constants and type definitions
used throughout the Genetic Algorithm music generation project. Consolidating
these values here improves maintainability, readability, and allows for easier
tuning of the algorithm's behavior and musical output.

It includes:
- Type Aliases: For clear and consistent data structure definitions (e.g., MelodySequence).
- Genetic Algorithm Parameters: Settings that control the evolutionary process
  (e.g., population size, mutation rate).
- Musical Definitions: Constants related to music theory and structure
  (e.g., pitch ranges, common rhythms, scale definitions).
- Fitness Function Weights: Coefficients that determine the importance of various
  musical criteria in evaluating generated melodies.
- Other Musical Constraints: Rules and preferences for melody generation
  (e.g., maximum leap intervals, phrase characteristics).
"""

from typing import List, Tuple, Optional, Dict, TypedDict, Any

# --- Type Aliases for Clarity ---

# Represents a single musical event: a pitch (string like "C4" or None for a rest)
# and its duration in quarter notes (float).
MelodyNote = Tuple[Optional[str], float]

# Represents a sequence of musical events (notes and rests), forming a melody or phrase.
MelodySequence = List[MelodyNote]

# Defines the structure for storing analyzed information about a reference musical phrase.
# This is used to guide the generation process and for fitness evaluation.
class ReferencePhraseInfo(TypedDict):
    """
    A TypedDict representing the analyzed characteristics of a musical phrase.

    Attributes:
        start_index (int): The starting index of this phrase within the original full melody sequence.
        end_index (int): The ending index of this phrase within the original full melody sequence.
        num_notes (int): The number of musical events (notes or rests) in this phrase.
        duration_beats (float): The total duration of the phrase in beats.
        character (str): The analyzed character of the phrase (e.g., "question", "answer", "neutral").
        notes (MelodySequence): The sequence of notes and rests constituting the phrase.
        # Potentially: start_tonal_function, end_tonal_function (if fully implemented)
    """
    start_index: int
    end_index: int
    num_notes: int
    duration_beats: float
    character: str
    notes: MelodySequence
    # The following were in the original PhraseInfo but not used in current analysis logic.
    # Retained as comments for potential future expansion.
    # start_tonal_function: Optional[str]
    # end_tonal_function: Optional[str]


# --- Genetic Algorithm Parameters ---
POPULATION_SIZE: int = 50  # Number of individuals (melodies) in each generation.
N_GENERATIONS: int = 50    # Total number of generations for the evolution (if running a fixed loop).
MUTATION_RATE: float = 0.10 # Probability of a mutation occurring in an individual's gene (note/duration).
CROSSOVER_RATE: float = 0.7 # Probability of two parents producing offspring.
TOURNAMENT_SIZE: int = 5   # Number of individuals selected for a tournament in tournament selection.

# --- Musical Definitions ---
MIN_OCTAVE: int = 3        # Minimum octave for generated notes (e.g., C3).
MAX_OCTAVE: int = 5        # Maximum octave for generated notes (e.g., B5).
# Common rhythmic durations in quarter notes.
POSSIBLE_DURATIONS: List[float] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0] # Shortened for more variety. Added 0.75. Removed 3.0, 4.0.

# Definitions of common scales by their interval patterns from the root (in semitones).
SCALE_TYPES: Dict[str, List[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],       # W-W-H-W-W-W-H
    "natural_minor": [0, 2, 3, 5, 7, 8, 10], # W-H-W-W-H-W-W
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],# W-H-W-W-H-WH-H
    "melodic_minor_asc": [0, 2, 3, 5, 7, 9, 11], # W-H-W-W-W-W-H (ascending)
}
# Mapping of key names to their root note's MIDI pitch value (enharmonics considered).
# Note: music21 handles key parsing robustly, this is more for reference or specific utilities.
KEY_SIGNATURES: Dict[str, int] = { # Root note MIDI value for C as 0
    "C": 0, "G": 7, "D": 2, "A": 9, "E": 4, "B": 11, "F#": 6, "C#": 1,
    "F": 5, "Bb": 10, "Eb": 3, "Ab": 8, "Db": 1, "Gb": 6, "Cb": 11
}

# --- Fitness Function Weights ---
# These weights control the importance of different musical criteria when evaluating
# the "goodness" of a generated melody. Higher values mean stronger influence.

# Reference-based scores (comparing to an input MIDI, if provided)
WEIGHT_REF_PITCH_EXACT: float = 2.0         # Bonus for matching reference pitch exactly.
WEIGHT_REF_PITCH_CLOSE: float = 1.0         # Bonus for being close (e.g., +/- 2 semitones) to reference pitch.
WEIGHT_REF_PITCH_PENALTY_FACTOR: float = 1.5# Multiplier for penalty if pitch is very different.
WEIGHT_REF_RHYTHM_EXACT: float = 2.5        # Bonus for matching reference rhythm exactly.
WEIGHT_REF_RHYTHM_CLOSE: float = 1.2        # Bonus for being close to reference rhythm (e.g., within 25%).
WEIGHT_REF_CONTOUR_SIMILARITY: float = 0.7  # Bonus for matching the general melodic contour of the reference.
WEIGHT_REF_QA_CHARACTER_MATCH: float = 1.5  # Bonus for matching Question/Answer phrase character of reference.
WEIGHT_REF_QA_PHRASE_NOTE_SIMILARITY: float = 1.0 # Bonus for note similarity within matched Q/A phrases.

# Intrinsic musicality scores (evaluating general musical qualities)
WEIGHT_PHRASE_END_LONG_NOTE_OR_REST: float = 0.8 # Bonus for phrases ending on a longer note or rest.
WEIGHT_PENALTY_MID_PHRASE_LONG_NOTE: float = -0.6 # Penalty for overly long notes in the middle of phrases.
WEIGHT_MELODY_RANGE_PENALTY: float = -0.7   # Penalty for melodies exceeding a preferred compact range.
WEIGHT_SCALE_DEGREE_RESOLUTION: float = 0.9 # Bonus for notes resolving according to tonal functions (e.g., leading tone to tonic).
WEIGHT_STEPWISE_MOTION: float = 0.6         # Bonus for smoother, stepwise melodic motion.
MIN_STEPWISE_MOTION_RATIO: float = 0.3      # Minimum desired ratio of stepwise movements in a melody.
WEIGHT_LARGE_LEAP_PENALTY: float = 0.8      # Penalty factor for large melodic leaps (positive value, applied as subtraction).
WEIGHT_OUT_OF_KEY_PENALTY: float = -2.0     # Strong penalty for notes not belonging to the current key.
WEIGHT_EXCESSIVE_REPEATED_PITCH_PENALTY: float = 0.5 # Penalty factor for too many consecutive identical pitches.
WEIGHT_INTERNAL_PHRASE_REPETITION: float = 0.7 # Bonus for some degree of motivic repetition/development within the melody.

# --- Melody Generation Parameters ---
NOTES_PER_PHRASE_FALLBACK: int = 8 # Approximate number of notes per phrase if not guided by reference phrase durations.
DEFAULT_PHRASE_DURATION_BEATS: float = 4.0 # Target duration for phrases in beats (e.g., 1 measure in 4/4 time).
MAX_CONSECUTIVE_REPEATED_PITCHES: int = 3 # Maximum allowed consecutive identical pitches before penalty or change is encouraged.
MOTIF_REPETITION_CHANCE: float = 0.25      # Probability of attempting to repeat/develop a previous motif.
INTERNAL_PHRASE_REPETITION_CHANCE: float = 0.35 # Probability of developing a previous phrase.
PHRASE_PUNCTUATION_CHANCE: float = 0.6     # Probability of applying punctuation (e.g., lengthening the last note) to a phrase.
REST_PROBABILITY_GENERATION: float = 0.12  # Probability of generating a rest instead of a note.
REST_PROBABILITY_MUTATION: float = 0.05    # Probability of a note mutating into a rest.
PHRASE_END_MUTATION_CHANCE: float = 0.2    # Probability of mutating the end of a phrase specifically (e.g., duration).

# --- Other Musical Constraints ---
TARGET_MELODY_RANGE_SEMITONES: int = 16  # Preferred comfortable melodic range (e.g., just over two octaves).
MAX_ALLOWED_MELODY_RANGE_SEMITONES: int = 24 # Absolute maximum melodic range (two octaves).
MAX_LEAP_INTERVAL: int = 12             # Max allowed melodic leap in semitones (octave). Discouraged.
ACCEPTABLE_LEAP_INTERVAL: int = 7       # Max leap considered acceptable without specific justification (Perfect 5th).
MIN_NOTES_PER_PHRASE: int = 2           # Minimum number of notes to constitute a phrase.

# --- Default values if no reference MIDI is provided ---
DEFAULT_KEY: str = "C major"            # Default key signature for generation.
DEFAULT_SCALE_TYPE: str = "major"       # Default scale type (used if key string doesn't specify mode).
DEFAULT_NUM_MEASURES: int = 4           # Default length in measures if generating without reference.

# --- Phrase Analysis Constants (used in music_utils) ---
# These help in determining phrase characteristics like "question" or "answer".
QUESTION_ENDING_NOTES_STABILITY: List[str] = ["dominant", "supertonic", "leading_tone"] # Scale degrees often ending questions.
ANSWER_ENDING_NOTES_STABILITY: List[str] = ["tonic", "mediant"]       # Scale degrees often ending answers.
SHORT_PHRASE_DURATION_THRESHOLD: float = 2.0 # Phrases shorter than this in beats might be treated differently.
LONG_NOTE_DURATION_THRESHOLD: float = 1.5    # Notes longer than this in beats are considered "long".

# For analyzing tonal functions based on scale degrees (0-11 semitones from tonic).
TONAL_FUNCTION_MAP: Dict[int, str] = {
    0: "Tonic", 1: "Supertonic_b2", 2: "Supertonic", 3: "Mediant_b3", 4: "Mediant",
    5: "Subdominant", 6: "Tritone/Subdominant_#4", # Corrected for clarity
    7: "Dominant", 8: "Submediant_b6", 9: "Submediant",
    10: "Subtonic/Dominant_b7", 11: "Leading Tone"
}

# Print statement to confirm module loading, useful for debugging.
# Can be commented out in production.
print(f"--- music_constants.py (version {__import__('datetime').date.today()}) loaded ---")