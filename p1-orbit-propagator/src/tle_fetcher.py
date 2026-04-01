import requests
import json
from pathlib import Path

TARGETS = {
    "ISS":          25544,
    "Hubble":       20580,
    "Starlink-30":  55765,
    "NOAA-20":      43013,
}

def fetch_tle(norad_id, cache_dir=Path("data")):
    """
    Downloads orbital elements in OMM (JSON) format from CelesTrak.
    """
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{norad_id}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=JSON"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()[0]

    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)

    return data

def fetch_all_targets():
    return {name: fetch_tle(norad_id) for name, norad_id in TARGETS.items()}