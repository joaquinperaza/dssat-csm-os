"""Generate a synthetic weather JSON for testing.

Run once::

    python tests/generate_example_weather.py
"""

import json
import math
import random
from pathlib import Path

random.seed(42)

lat = 29.63
lon = -82.37
elev = 30.0
tav = 22.5
tamp = 7.8
start_year = 1982
start_doy = 100
n_days = 250

records = []
for i in range(n_days):
    doy = start_doy + i
    year = start_year
    while doy > 365:
        doy -= 365
        year += 1
    yrdoy = year * 1000 + doy

    # Simple sinusoidal seasonal climate
    angle = 2 * math.pi * (doy - 15) / 365.0
    tmax = tav + tamp * math.sin(angle) + random.gauss(0, 2)
    tmin = tmax - 10.0 + random.gauss(0, 1.5)
    srad = max(5, 18 + 8 * math.sin(angle) + random.gauss(0, 2))
    rain = max(0.0, random.expovariate(0.1) - 5.0) if random.random() < 0.3 else 0.0

    records.append({
        "date": yrdoy,
        "tmax": round(tmax, 1),
        "tmin": round(tmin, 1),
        "srad": round(srad, 1),
        "rain": round(rain, 1),
        "windsp": round(random.uniform(100, 300), 0),
        "rhum": round(random.uniform(50, 90), 0),
    })

data = {
    "lat": lat,
    "lon": lon,
    "elev": elev,
    "tav": tav,
    "tamp": tamp,
    "records": records,
}

out_path = Path(__file__).parent / "example_weather.json"
with out_path.open("w") as f:
    json.dump(data, f, indent=2)

print(f"Written {len(records)} records to {out_path}")
