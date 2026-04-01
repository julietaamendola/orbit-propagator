import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from src.tle_fetcher import fetch_all_targets
from src.propagator  import (propagate_sgp4, propagate_keplerian,
                              propagate_j2, propagate_j2_drag,
                              get_ground_track)
from src.visualizer  import (plot_ground_track, plot_orbital_elements_evolution,
                              plot_propagator_comparison)

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
    positions_24h, velocities_24h, _ = propagate_sgp4(
        iss_omm, t_start, duration_hours=24
    )
    plot_orbital_elements_evolution(iss_omm, t_start, positions_24h,
                                    title="ISS — 24 hours (SGP4)")

    # --- propagator comparison: 4 models vs SGP4 over 7 days ---
    print("\nComparing propagators over 7 days...")
    positions_sgp4, velocities_sgp4, _ = propagate_sgp4(
        iss_omm, t_start, duration_hours=168
    )
    r0 = positions_sgp4[0]
    v0 = velocities_sgp4[0]

    print("  Keplerian...")
    positions_keplerian = propagate_keplerian(r0, v0, duration_hours=168)
    print("  RK4 + J2...")
    positions_j2        = propagate_j2(r0, v0, duration_hours=168)
    print("  RK4 + J2 + drag...")
    positions_j2_drag   = propagate_j2_drag(r0, v0, duration_hours=168)

    plot_propagator_comparison(
        {
            "SGP4":            positions_sgp4,
            "Keplerian":       positions_keplerian,
            "RK4 + J2":        positions_j2,
            "RK4 + J2 + drag": positions_j2_drag,
        },
        title="ISS — propagator hierarchy vs SGP4 reference (7 days)"
    )

    print("\nDone. Plots saved to /plots/")


if __name__ == "__main__":
    main()