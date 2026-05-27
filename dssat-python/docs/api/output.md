# Output

`Simulation.run()` returns a dict with two keys:

## `results["summary"]`

End-of-season scalar values collected by `CropModel.summary()`:

| Key | Unit | Description |
|-----|------|-------------|
| `yield_kg_ha` | kg ha⁻¹ | Grain (or storage organ) yield |
| `biomas_g_m2` | g m⁻² | Total above-ground dry matter at harvest |
| `peak_lai` | m² m⁻² | Maximum leaf area index |
| `das_maturity` | days | Days from planting to physiological maturity |
| `das_silking` | days | Days from planting to silking (CERES-Maize only) |
| `harvest_date` | YYYYDDD | Harvest date |
| `swfac_min` | – | Minimum water stress factor during season (0=max stress) |
| `nstres_min` | – | Minimum N stress factor |

## `results["daily"]`

A `pandas.DataFrame` with one row per simulated day:

| Column | Unit | Description |
|--------|------|-------------|
| `date` | YYYYDDD | Calendar date |
| `yrdoy` | YYYYDDD | Same as `date` |
| `das` | days | Days after sowing |
| `tmax` | °C | Maximum air temperature |
| `tmin` | °C | Minimum air temperature |
| `srad` | MJ m⁻² | Daily solar radiation |
| `rain` | mm | Precipitation |
| `tsw_cm` | cm | Total soil water (root zone) |
| `runoff_mm` | mm | Surface runoff |
| `drain_mm` | mm | Deep drainage |
| `snow_mm` | mm | Snow water equivalent |
| `istage` | 1–9 | Integer growth stage |
| `xstage` | – | Fractional growth stage |
| `lai` | m² m⁻² | Leaf area index |
| `biomas_g_m2` | g m⁻² | Cumulative above-ground biomass |
| `yield_kg_ha` | kg ha⁻¹ | Cumulative grain yield |
| `swfac` | 0–1 | Water stress factor |
| `nstres` | 0–1 | Nitrogen stress factor |
