"""Two-way converters between DSSAT legacy file formats and JSON.

Supported formats:

- **Weather** — ``.WTH`` ↔ :mod:`~dssat.io.converters.wth`
- **Soil** — ``.SOL`` ↔ :mod:`~dssat.io.converters.sol`
- **Experiment** — ``.MZX / .WHX / .SBX / …`` ↔ :mod:`~dssat.io.converters.xfile`
- **Cultivar** — ``.CUL`` ↔ :mod:`~dssat.io.converters.cul`
- **Ecotype** — ``.ECO`` ↔ :mod:`~dssat.io.converters.eco`
- **Species** — ``.SPE`` ↔ :mod:`~dssat.io.converters.spe`

Quick usage::

    from dssat.io.converters import wth, sol, xfile, cul, eco, spe

    # WTH → JSON
    weather_json = wth.read_wth("UFGA8201.WTH")

    # JSON → WTH
    wth.write_wth(weather_json, "UFGA8201.WTH")

    # SOL → JSON list
    profiles = sol.read_sol("SOIL.SOL")

    # X-file → JSON
    exp = xfile.read_xfile("UFGA8201.MZX")

    # JSON → X-file
    xfile.write_xfile(exp, "UFGA8201.MZX")

    # ECO → JSON
    ecotypes = eco.read_eco("MZCER048.ECO")
    json_str = eco.eco_to_json("MZCER048.ECO")

    # JSON → ECO
    eco.json_to_eco(json_str, "MZCER048_out.ECO")

    # SPE → JSON
    species = spe.read_spe("MZCER048.SPE")
    json_str = spe.spe_to_json("MZCER048.SPE")

    # JSON → SPE
    spe.json_to_spe(json_str, "MZCER048_out.SPE")
"""

from dssat.io.converters import sol, wth, xfile, cul, eco, spe

__all__ = ["wth", "sol", "xfile", "cul", "eco", "spe"]
