"""JSON I/O layer — readers, writers and DSSAT legacy file converters.

JSON readers
------------
:func:`load_experiment`, :func:`load_weather`, :func:`load_soil`,
:func:`build_initial_sw`, :func:`build_initial_nutrients`

Legacy DSSAT converters
-----------------------
:mod:`dssat.io.converters.wth`  — ``.WTH`` weather files
:mod:`dssat.io.converters.sol`  — ``.SOL`` soil files
:mod:`dssat.io.converters.xfile` — experiment files (``.MZX``, ``.WHX``, …)
"""

from dssat.io.readers import (
    load_experiment,
    load_weather,
    load_soil,
    build_initial_sw,
    build_initial_nutrients,
)
from dssat.io import converters

__all__ = [
    "load_experiment",
    "load_weather",
    "load_soil",
    "build_initial_sw",
    "build_initial_nutrients",
    "converters",
]
