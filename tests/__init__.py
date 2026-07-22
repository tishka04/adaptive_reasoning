"""Project-local test helpers.

Keeping this directory an explicit package prevents unrelated third-party
``tests`` packages from shadowing intra-suite helper imports.
"""

from pathlib import Path
import sys


_TESTS_DIRECTORY = str(Path(__file__).resolve().parent)
if _TESTS_DIRECTORY not in sys.path:
    sys.path.insert(0, _TESTS_DIRECTORY)
