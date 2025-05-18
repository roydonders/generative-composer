# ga_components/__init__.py

"""
Genetic Algorithm Components Package

This package contains modules for a genetic algorithm designed to generate musical melodies.
It includes:
- music_constants: Defines core data structures (Note, MelodySequence, PhraseInfo)
                   and musical/algorithmic constants.
- music_utils: Provides utility functions for music theory calculations, MIDI parsing,
               and analysis (e.g., key detection, scale analysis, phrase characterization).
- melody_generator: Handles the creation of initial melodies and their mutation and
                    crossover during the evolutionary process.
- genetic_algorithm_core: Contains the main GeneticAlgorithm class, orchestrating
                          the evolutionary loop, including fitness calculation,
                          selection, and population management.
"""

# You can choose to expose specific classes or functions at the package level for convenience.
# For example, if you frequently import MelodyGenerator or GeneticAlgorithm:
#
# from .melody_generator import MelodyGenerator
# from .genetic_algorithm_core import GeneticAlgorithm
# from .music_utils import MusicUtils
# from .music_constants import Note, MelodySequence, PhraseInfo
#
# This allows imports like: from ga_components import MelodyGenerator
#
# However, for clarity in larger projects, explicit imports from submodules are often preferred,
# e.g., from ga_components.melody_generator import MelodyGenerator.
# For this project, we'll keep the __init__.py simple, and the main scripts
# (backend_logic.py) will import directly from the submodules.

# This print statement can be useful for debugging to confirm the package is being loaded.
# print("ga_components package loaded.")

# Version of the package (optional)
__version__ = "1.0.0"