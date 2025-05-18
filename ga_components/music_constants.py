# ga_components/music_constants.py
"""
Module: music_constants.py

Purpose:
This module serves as a central repository for all constants, type definitions,
and configuration parameters used throughout the Genetic Algorithm (GA) for music
generation. Consolidating these values here enhances maintainability, readability,
and allows for easier tuning of the algorithm's behavior and the musical
characteristics of the generated output.

Key Sections:
- Type Aliases: Define custom types for clarity and type checking (e.g., MelodySequence).
- Genetic Algorithm Parameters: Control the core evolutionary process (e.g., population size).
- Musical Definitions: Basic musical parameters like octave ranges and rhythmic units.
- Fitness Function Weights: Crucial for guiding the evolution. These weights determine
  the relative importance of various musical criteria when evaluating melodies.
  They are categorized into reference-matching and intrinsic musicality.
- Melody Generation Parameters: Control aspects of how new melodies are initially
  created and mutated by the MelodyGenerator.
- Other Musical Constraints: Define acceptable limits for musical features.
"""

from typing import List, Tuple, Optional, Dict, TypedDict, Any

# --- Type Aliases for Clarity ---

# MelodyNote: Represents a single musical event.
# It's a tuple containing:
#   - Optional[str]: The pitch of the note as a string (e.g., "C4", "F#5").
#                    None if the event is a rest.
#   - float: The duration of the event in quarter notes (e.g., 1.0 for a quarter note, 0.5 for an eighth).
MelodyNote = Tuple[Optional[str], float]

# MelodySequence: Represents a sequence of musical events (notes and rests),
# effectively defining a melody or a musical phrase.
MelodySequence = List[MelodyNote]

# ReferencePhraseInfo: A TypedDict defining the structure for storing analyzed
# information about a musical phrase, whether from a reference MIDI or an evolved melody.
# This structure is used for comparative analysis in the fitness function.
class ReferencePhraseInfo(TypedDict):
    """
    TypedDict representing the analyzed characteristics of a musical phrase.

    Attributes:
        start_index (int): The starting index of this phrase within the original full melody sequence.
        end_index (int): The ending index (inclusive) of this phrase.
        num_notes (int): The number of musical events (notes or rests) in this phrase.
        duration_beats (float): The total duration of the phrase in beats (sum of quarter lengths).
        character (str): Analyzed character (e.g., "question", "answer", "neutral").
                         (Analysis performed by MusicUtils, can be used for advanced fitness).
        notes (MelodySequence): The sequence of MelodyNote tuples constituting the phrase.
    """
    start_index: int
    end_index: int
    num_notes: int
    duration_beats: float
    character: str
    notes: MelodySequence

# --- Genetic Algorithm Parameters ---
POPULATION_SIZE: int = 5       # Number of individuals (melodies) in each generation.
                                # Reduced for faster evolution cycles.
MUTATION_RATE: float = 0.15     # Probability of a gene (pitch/duration) within an individual undergoing mutation.
CROSSOVER_RATE: float = 0.80    # Probability that two selected parent melodies will recombine to produce offspring.
TOURNAMENT_SIZE: int = 4        # Number of individuals randomly selected for a tournament;
                                # the fittest among them becomes a parent. Adjusted for smaller population.

# --- Musical Definitions ---
MIN_OCTAVE: int = 3             # Default minimum octave for generated notes (e.g., C3).
MAX_OCTAVE: int = 5             # Default maximum octave for generated notes (e.g., B5).
# Common rhythmic durations in quarter notes, used by MelodyGenerator.
POSSIBLE_DURATIONS: List[float] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

# --- Fitness Function Weights (Balanced: Reference Focus + Musicality) ---
# These weights are critical. Higher positive values indicate stronger rewards,
# while negative values indicate penalties. The magnitude reflects importance.

# Category 1: Core Reference Matching (Highest Priority)
# These ensure the evolved melody structurally and fundamentally resembles the reference.
WEIGHT_REF_EVENT_COUNT_MATCH: float = 6.0   # Strongest emphasis: matching the number of notes/rests.
WEIGHT_REF_TOTAL_DURATION_MATCH: float = 6.0 # Strongest emphasis: matching total duration in seconds.
WEIGHT_REF_PITCH_EXACT: float = 4.5         # Strong bonus for matching reference pitch exactly.
WEIGHT_REF_RHYTHM_EXACT: float = 4.5        # Strong bonus for matching reference rhythm exactly.

# Category 2: Phrase Boundary Matching (High Priority)
# Ensures key structural points of phrases align with the reference.
WEIGHT_REF_PHRASE_START_PITCH_MATCH: float = 3.5 # Bonus for matching the first pitch of a corresponding phrase.
WEIGHT_REF_PHRASE_END_PITCH_MATCH: float = 3.5   # Bonus for matching the last pitch of a corresponding phrase.

# Category 3: Penalties for Gross Mismatches with Reference (High Impact)
PENALTY_REF_PITCH_MISMATCH: float = -2.5    # Applied when both are notes but pitches differ.
PENALTY_REF_RHYTHM_MISMATCH: float = -2.5   # Applied when durations differ significantly.
PENALTY_REF_NOTE_VS_REST_MISMATCH: float = -3.5 # Applied when one is a note and the other is a rest.

# Category 4: Intrinsic Musicality & Coherence (Moderate Influence)
# These guide the melody towards generally "good" musical practices, especially
# when reference matching scores are similar or for regions not strictly defined by the reference.
PENALTY_OUT_OF_KEY: float = -2.0            # Penalty for notes not belonging to the current key.
WEIGHT_SCALE_DEGREE_RESOLUTION: float = 1.0 # Rewards common tonal resolutions (e.g., 7->1).
WEIGHT_STEPWISE_MOTION: float = 0.8         # Rewards smooth, stepwise melodic motion.
MIN_STEPWISE_MOTION_RATIO: float = 0.35     # Target ratio of stepwise movements to total movements.
PENALTY_LARGE_LEAPS: float = -1.5           # Penalty for large melodic leaps (applied if interval > ACCEPTABLE_LEAP_INTERVAL).
WEIGHT_MELODY_RANGE_PENALTY: float = -1.0   # Penalty for melodies exceeding a preferred compact range.
TARGET_MELODY_RANGE_SEMITONES: int = 18     # Preferred melodic range (approx 1.5 octaves).
MAX_ALLOWED_MELODY_RANGE_SEMITONES: int = 28 # Harder limit for range penalty application.
PENALTY_EXCESSIVE_REPETITION: float = -1.0  # Penalty for too many consecutive identical pitches.
WEIGHT_MELODIC_CONTOUR_SIMILARITY: float = 1.0 # Bonus for matching the general up/down contour of reference phrases.

# --- Melody Generation Parameters (for MelodyGenerator.py) ---
# These influence how the MelodyGenerator creates and mutates melodies.
NOTES_PER_PHRASE_FALLBACK: int = 8  # Approx notes per phrase if not guided by reference.
DEFAULT_PHRASE_DURATION_BEATS: float = 4.0 # Target phrase duration if not guided by reference.
MAX_CONSECUTIVE_REPEATED_PITCHES: int = 3 # Max allowed before PENALTY_EXCESSIVE_REPETITION applies.
REST_PROBABILITY_GENERATION: float = 0.08 # Chance of generating a rest instead of a note.
REST_PROBABILITY_MUTATION: float = 0.04   # Chance of a note mutating into a rest.
MOTIF_REPETITION_CHANCE: float = 0.10     # Reduced to limit deviation from reference.
INTERNAL_PHRASE_REPETITION_CHANCE: float = 0.15 # Internal phrase repetition chance. Reduced.
PHRASE_PUNCTUATION_CHANCE: float = 0.2    # Phrase punctuation chance. Reduced.
PHRASE_END_MUTATION_CHANCE: float = 0.05  # Phrase end mutation chance. Reduced.

# --- Phrase Analysis Constants ---
# These lists define which tonal functions (derived from scale degrees)
# are typically associated with "questioning" (less resolved) or "answering"
# (more resolved) phrase endings in traditional Western music theory.
QUESTION_ENDING_NOTES_STABILITY: List[str] = ["dominant", "supertonic", "leading_tone"]
ANSWER_ENDING_NOTES_STABILITY: List[str] = ["tonic", "mediant"]

# --- Other Musical Constraints ---
ACCEPTABLE_LEAP_INTERVAL: int = 7   # Max leap (semitones) considered acceptable without PENALTY_LARGE_LEAPS (Perfect 5th).
MIN_NOTES_PER_PHRASE: int = 1       # Minimum notes to constitute a phrase for analysis purposes.

# --- Default values for generation if no reference MIDI is provided ---
DEFAULT_KEY: str = "C major"        # Default key signature if none detected.

# Informative print statement upon module loading.
print(f"--- music_constants.py (Comprehensive Fitness, v. {__import__('datetime').date.today()}) loaded ---")
