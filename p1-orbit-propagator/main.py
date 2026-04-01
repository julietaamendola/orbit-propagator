import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from src.propagator import propagate_sgp4, get_ground_track
from src.visualizer import plot_ground_track, plot_orbital_elements_evolution
from src.tle_fetcher import fetch_all_targets

Path("plots").mkdir(exist_ok=True)


def main():
    print("Fetching TLEs from CelesTrak...")
    satellites = fetch_all_targets()

    t_start = datetime.now(timezone.utc)

    # --- ground tracks: 4.5 hours (~3 ISS orbits) ---
    tracks = {}
    for name, omm in satellites.items():
        print(f"  Propagating {name}...")
        try:
            lats, lons = get_ground_track(omm, t_start, duration_hours=4.5)
            tracks[name] = (np.array(lats), np.array(lons))
        except Exception as e:
            print(f"    Error with {name}: {e}")

    plot_ground_track(tracks, save_path=Path("plots/ground_tracks.png"))

    # --- orbital elements evolution: ISS over 24 hours ---
    print("\nPlotting orbital elements evolution...")
    iss_omm = satellites["ISS"]
    positions_24h, _, _ = propagate_sgp4(iss_omm, t_start, duration_hours=24)
    plot_orbital_elements_evolution(positions_24h,
                                    title="ISS — altitude oscillation over 24 hours (SGP4)")

    print("\nDone. Plots saved to /plots/")


if __name__ == "__main__":
    main()