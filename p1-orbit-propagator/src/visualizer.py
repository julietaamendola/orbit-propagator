import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

COLORS = {
    "ISS":         "#E8593C",
    "Hubble":      "#378ADD",
    "Starlink-30": "#1D9E75",
    "NOAA-20":     "#7F77DD",
}

PROPAGATOR_COLORS = {
    "Keplerian":       "#7F77DD",
    "RK4 + J2":        "#1D9E75",
    "RK4 + J2 + drag": "#FAC775",
}


def plot_ground_track(tracks, save_path=None):
    """
    Ground track map with natural earth background (cartopy),
    start/end markers, and dark style.
    Falls back to simple matplotlib if cartopy is not installed.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        _plot_ground_track_cartopy(tracks, save_path)
    except ImportError:
        _plot_ground_track_simple(tracks, save_path)


def _plot_ground_track_cartopy(tracks, save_path=None):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(16, 8), facecolor="#0d1117")
    ax  = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_facecolor("#0d1117")

    ax.add_feature(cfeature.OCEAN,     facecolor="#0d1f33", zorder=0)
    ax.add_feature(cfeature.LAND,      facecolor="#1a2a1a", zorder=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#ffffff30", linewidth=0.4, zorder=2)
    ax.add_feature(cfeature.BORDERS,   edgecolor="#ffffff15", linewidth=0.3, zorder=2)
    ax.gridlines(color="#ffffff10", linewidth=0.4, zorder=2)
    ax.set_global()

    for name, (lats, lons) in tracks.items():
        lats  = np.array(lats)
        lons  = np.array(lons)
        color = COLORS.get(name, "#ffffff")

        breaks   = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
        segments = np.split(np.column_stack([lons, lats]), breaks)

        for i, seg in enumerate(segments):
            ax.plot(seg[:, 0], seg[:, 1],
                    color=color, linewidth=1.2, alpha=0.85,
                    transform=ccrs.PlateCarree(),
                    label=name if i == 0 else None)

        ax.scatter(lons[0], lats[0], color=color, s=80, zorder=6,
                   marker="*", edgecolors="white", linewidths=0.6,
                   transform=ccrs.PlateCarree())
        ax.scatter(lons[-1], lats[-1], color=color, s=60, zorder=6,
                   marker="o", edgecolors="white", linewidths=0.6,
                   transform=ccrs.PlateCarree())

    sat_legend = ax.legend(loc="lower left", framealpha=0.2,
                           labelcolor="white", fontsize=9,
                           facecolor="#0d1117", edgecolor="#ffffff20")

    marker_legend = ax.legend(
        handles=[
            Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
                   markersize=8, label="Start", linestyle="None"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markersize=6, label="End",   linestyle="None"),
        ],
        loc="lower right", framealpha=0.2, labelcolor="white",
        fontsize=9, facecolor="#0d1117", edgecolor="#ffffff20"
    )
    ax.add_artist(sat_legend)

    ax.set_title("Ground tracks — 4.5 hours · SGP4 via Skyfield",
                 color="#ffffffcc", fontsize=11, pad=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()


def _plot_ground_track_simple(tracks, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    for lat in range(-90, 91, 30):
        ax.axhline(lat, color="#ffffff10", linewidth=0.4)
    for lon in range(-180, 181, 30):
        ax.axvline(lon, color="#ffffff10", linewidth=0.4)

    for name, (lats, lons) in tracks.items():
        lats  = np.array(lats)
        lons  = np.array(lons)
        color = COLORS.get(name, "#ffffff")

        breaks   = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
        segments = np.split(np.column_stack([lons, lats]), breaks)

        for i, seg in enumerate(segments):
            ax.plot(seg[:, 0], seg[:, 1], color=color,
                    linewidth=1.2, alpha=0.85,
                    label=name if i == 0 else None)

        ax.scatter(lons[0], lats[0], color=color, s=80, zorder=5,
                   marker="*", edgecolors="white", linewidths=0.8)
        ax.scatter(lons[-1], lats[-1], color=color, s=60, zorder=5,
                   marker="o", edgecolors="white", linewidths=0.8)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.tick_params(colors="#ffffff50", labelsize=8)
    ax.set_xlabel("Longitude", color="#ffffff70", fontsize=9)
    ax.set_ylabel("Latitude",  color="#ffffff70", fontsize=9)
    ax.legend(loc="lower left", framealpha=0.15,
              labelcolor="white", fontsize=9)
    ax.set_title("Ground tracks — 4.5 hours · SGP4 via Skyfield",
                 color="#ffffffcc", fontsize=11, pad=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()


def plot_orbital_elements_evolution(positions_km, title="Orbital elements evolution"):
    """
    Shows altitude oscillation over time.
    """
    r_norms   = np.linalg.norm(positions_km, axis=1)
    altitudes = r_norms - 6371
    t_hours   = np.arange(len(altitudes)) / 60

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_hours, altitudes, color="#1D9E75", linewidth=1.2)
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_propagator_comparison(positions_dict,
                               title="Propagator hierarchy — positional drift vs SGP4"):
    """
    Plots drift curves for multiple propagators against SGP4 as reference.
    positions_dict: {"label": positions_array, ...}
    SGP4 must be included under the key "SGP4".

    Four panels:
    - Top left:     raw error full 7 days
    - Top right:    zoom first 24 hours
    - Bottom left:  orbit-averaged secular trend
    - Bottom right: log scale full 7 days
    """
    ref = positions_dict["SGP4"]

    errors = {}
    for label, positions in positions_dict.items():
        if label == "SGP4":
            continue
        min_len       = min(len(ref), len(positions))
        diff          = ref[:min_len] - positions[:min_len]
        errors[label] = np.linalg.norm(diff, axis=1)

    n_steps = min(len(e) for e in errors.values())
    t_hours = np.arange(n_steps) / 60

    def orbit_avg(err, window=92):
        return np.convolve(err, np.ones(window)/window, mode='valid')

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle(title, fontsize=12)

    for label, err in errors.items():
        err   = err[:n_steps]
        color = PROPAGATOR_COLORS.get(label, "#aaaaaa")

        axes[0, 0].plot(t_hours, err, color=color, linewidth=0.6,
                        alpha=0.7, label=label)

        mask = t_hours <= 24
        axes[0, 1].plot(t_hours[mask], err[mask], color=color,
                        linewidth=1.2, label=label)

        err_avg = orbit_avg(err)
        t_avg   = t_hours[:len(err_avg)]
        axes[1, 0].plot(t_avg, err_avg, color=color,
                        linewidth=1.4, label=label)

        axes[1, 1].semilogy(t_hours, err, color=color, linewidth=0.6,
                            alpha=0.7, label=label)

    axes[0, 0].set_title("Raw error — 7 days")
    axes[0, 0].set_xlabel("Time [hours]")
    axes[0, 0].set_ylabel("Error vs SGP4 [km]")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].set_title("Zoom — first 24 hours")
    axes[0, 1].set_xlabel("Time [hours]")
    axes[0, 1].set_ylabel("Error vs SGP4 [km]")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].set_title("Orbit-averaged error (secular trend)")
    axes[1, 0].set_xlabel("Time [hours]")
    axes[1, 0].set_ylabel("Mean error vs SGP4 [km]")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].set_title("Log scale — 7 days")
    axes[1, 1].set_xlabel("Time [hours]")
    axes[1, 1].set_ylabel("Error vs SGP4 [km] (log)")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.show()