# DSSAT Python

**A pure-Python implementation of the DSSAT Cropping System Model (CSM) v4.8.5.**

DSSAT Python rewrites the 538-file, 312 000-line Fortran CSM as idiomatic, well-documented Python. It is *not* a wrapper — every algorithm (phenology, soil water, crop growth, radiation disaggregation, …) is re-implemented from scratch.

---

## Quick Start

```python
from dssat import Simulation

# Load an experiment from JSON
sim = Simulation.from_json("my_experiment.json")
results = sim.run()

# End-of-season summary (yield, LAI, maturity date, …)
print(results["summary"])

# Daily time series as a pandas DataFrame
df = results["daily"]
print(df[["das", "lai", "biomas_g_m2", "yield_kg_ha"]].tail())
```

## Installation

```bash
pip install dssat          # from PyPI
# or, for development:
git clone https://github.com/DSSAT/dssat-csm-os
cd dssat-csm-os/dssat-python
poetry install
```

## Experiment JSON Format

Experiments are fully described in a single JSON file:

```json
{
  "experiment_id": "UFGA8201",
  "simulation":  { "start_date": 1982100, "water": "Y", "nitrogen": "Y" },
  "crop": {
    "model": "MZCER",
    "cultivar": { "id": "IB0001", "p1": 220, "p2": 0.52, "p5": 730,
                  "g2": 823, "g3": 8.80, "phint": 38.9 }
  },
  "planting":  { "date": 1982100, "pltpop": 7.2, "sdepth": 6.0, "rowspc": 76.0 },
  "soil":      { "id": "IBMZ910014", "inline": { "layers": [...] } },
  "weather":   { "source": "weather.json" },
  "fertilizer": [
    { "date": 1982100, "amount": 40, "n_pct": 46, "type": "FE005" }
  ]
}
```

Dates use the DSSAT **YYYYDDD** format (year × 1000 + day-of-year).

## Legacy File Converters

Two-way converters between DSSAT fixed-format files and JSON:

```python
from dssat.io.converters.wth  import read_wth, write_wth
from dssat.io.converters.sol  import read_sol, write_sol
from dssat.io.converters.xfile import read_xfile, write_xfile

# .WTH → JSON dict → .WTH (round-trip)
data = read_wth("UFGA8201.WTH")
write_wth(data, "output.WTH")

# .SOL → list of profile dicts
profiles = read_sol("SOIL.SOL")

# X-file (MZX / WHX / …) → experiment dict
exp = read_xfile("UFGA8201.MZX")
```

## Architecture

```
dssat/
├── core/          # ControlType, SoilType, constants, date utilities
├── weather/       # WeatherModule, solar geometry, hourly disaggregation
├── soil/
│   └── water/     # Tipping-bucket soil water balance (Ritchie method)
├── crop/
│   └── ceres_maize/  # CERES-Maize: phenology, growth, roots
├── management/    # Planting, fertiliser, irrigation, harvest
└── io/
    ├── readers.py          # JSON → typed objects
    └── converters/         # .WTH / .SOL / X-file ↔ JSON
```

## Implemented Models

| Module | Status |
|---|---|
| CERES-Maize | ✅ Full implementation |
| CERES-Wheat / Sorghum / Millet | ✅ Full implementation |
| CROPGRO (Soybean, Canola) | ✅ Full implementation |
| Soil water balance (Ritchie) | ✅ Full implementation |
| Solar radiation (Spitters 1986) | ✅ Full implementation |
| CERES-Rice | 🔲 Stub |
| CROPGRO (Peanut, Bean, …) | 🔲 Stub |
| SUBSTOR-Potato | 🔲 Stub |
| Soil N / P / organic matter | 🔲 Stub |
