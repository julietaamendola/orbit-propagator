import numpy as np
from datetime import timedelta
from sgp4.api import Satrec, jday
from sgp4 import omm as sgp4_omm

# Constants
MU = 3.986004418e14   # m³/s² — Earth gravitational parameter
RE = 6.371e6          # m     — Earth mean radius
J2 = 1.08263e-3       #       — J2 oblateness coefficient

# Atmospheric drag constants
CD  = 2.2    #       — drag coefficient (typical satellite)
A_M = 0.01   # m²/kg — area-to-mass ratio (typical ~3U CubeSat)

# Exponential atmosphere layers: (base altitude km, base density kg/m³, scale height km)
_ATMO_LAYERS = [
    (0,    1.225,      8.44),
    (25,   3.899e-2,   6.49),
    (30,   1.774e-2,   6.75),
    (40,   3.972e-3,   7.47),
    (50,   1.057e-3,   8.38),
    (60,   3.206e-4,   7.71),
    (70,   8.770e-5,   6.34),
    (80,   1.905e-5,   5.80),
    (90,   3.396e-6,   5.53),
    (100,  5.297e-7,   5.70),
    (200,  2.789e-10,  7.36),
    (300,  1.916e-12,  9.28),
    (400,  5.507e-13,  7.57),
    (500,  3.096e-13,  8.50),
    (600,  8.197e-14,  7.80),
    (700,  1.780e-14,  6.70),
    (800,  3.600e-15,  6.10),
    (900,  9.400e-16,  5.80),
    (1000, 3.019e-16,  5.40),
]

def compute_raan_over_time(omm, t_start, duration_hours, dt_seconds=60.0):
    """
    Computes true RAAN over time.
    Uses sgp4 for r,v in TEME, then rotates to GCRS using Skyfield's
    precession/nutation matrices for correct RAAN calculation.
    Returns t_hours, raan_deg arrays.
    """
    from sgp4.api import Satrec, jday
    from sgp4 import omm as sgp4_omm
    from skyfield.api import load
    from skyfield.positionlib import ITRF_to_GCRS2

    ts  = load.timescale()
    sat = Satrec()
    sgp4_omm.initialize(sat, omm)

    t_hours_list, raans = [], []
    t     = t_start
    hours = 0.0
    Z     = np.array([0.0, 0.0, 1.0])

    while hours <= duration_hours:
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute,
                      t.second + t.microsecond / 1e6)
        e, r_teme, v_teme = sat.sgp4(jd, fr)

        if e == 0:
            r = np.array(r_teme)
            v = np.array(v_teme)

            # Rotate TEME -> GCRS using Skyfield's time object
            # which has full precession/nutation corrections
            skyfield_t = ts.from_datetime(t)
            # Get the TEME->GCRS rotation matrix
            # Skyfield's EarthSatellite internally uses:
            # r_gcrs = skyfield_t.M @ r_teme  (where M is the rotation matrix)
            # We access it via the _time object internals
            r_gcrs = skyfield_t.M.T @ r
            v_gcrs = skyfield_t.M.T @ v

            # Compute RAAN from r_gcrs, v_gcrs in GCRS (inertial frame)
            h = np.cross(r_gcrs, v_gcrs)
            N = np.cross(Z, h)
            N_norm = np.linalg.norm(N)

            if N_norm > 1e-10:
                N_hat = N / N_norm
                raan  = np.degrees(np.arctan2(N_hat[1], N_hat[0]))
                raans.append(raan)
                t_hours_list.append(hours)

        t     += timedelta(seconds=dt_seconds)
        hours += dt_seconds / 3600

    return np.array(t_hours_list), np.array(raans)

# ── SGP4 ──────────────────────────────────────────────────────────────────────

def propagate_sgp4(omm, t_start, duration_hours, dt_seconds=60.0):
    """
    Propagates with SGP4 using OMM data from CelesTrak.
    Returns ECI positions [km], velocities [km/s], and times [datetime].
    """
    sat = Satrec()
    sgp4_omm.initialize(sat, omm)

    times, positions, velocities = [], [], []
    t = t_start

    while t <= t_start + timedelta(hours=duration_hours):
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute,
                      t.second + t.microsecond / 1e6)
        e, r, v = sat.sgp4(jd, fr)
        if e == 0:
            times.append(t)
            positions.append(r)
            velocities.append(v)
        t += timedelta(seconds=dt_seconds)

    return np.array(positions), np.array(velocities), np.array(times)


# ── Acceleration models ───────────────────────────────────────────────────────

def _accel_keplerian(r_vec):
    """Two-body gravity only. Input in meters, returns m/s²."""
    r = np.linalg.norm(r_vec)
    return -MU * r_vec / r**3


def _accel_j2(r_vec):
    """Two-body gravity + J2 oblateness. Input in meters, returns m/s²."""
    x, y, z = r_vec
    r  = np.linalg.norm(r_vec)
    r2 = r**2
    r5 = r**5
    k  = (3/2) * J2 * MU * RE**2 / r5

    ax = -MU * x / r**3 + k * x * (5*z**2/r2 - 1)
    ay = -MU * y / r**3 + k * y * (5*z**2/r2 - 1)
    az = -MU * z / r**3 + k * z * (5*z**2/r2 - 3)

    return np.array([ax, ay, az])


def _atmospheric_density(r_vec):
    """
    Exponential atmosphere model.
    Input r_vec in meters, returns density in kg/m³.
    """
    alt_km = (np.linalg.norm(r_vec) - RE) / 1e3

    if alt_km > 1000:
        return 0.0

    layer_alt, rho0, H = _ATMO_LAYERS[0]
    for i in range(len(_ATMO_LAYERS) - 1):
        if _ATMO_LAYERS[i][0] <= alt_km < _ATMO_LAYERS[i+1][0]:
            layer_alt, rho0, H = _ATMO_LAYERS[i]
            break

    return rho0 * np.exp(-(alt_km - layer_alt) / H)


def _accel_j2_drag(r_vec, v_vec):
    """
    Two-body gravity + J2 + atmospheric drag.
    Input in meters and m/s, returns m/s².
    """
    a     = _accel_j2(r_vec)
    rho   = _atmospheric_density(r_vec)
    v_mag = np.linalg.norm(v_vec)

    if v_mag > 0 and rho > 0:
        a += -0.5 * CD * A_M * rho * v_mag**2 * (v_vec / v_mag)

    return a


# ── Generic RK4 integrator ────────────────────────────────────────────────────

def _rk4_step(r, v, dt, accel_func):
    """
    Single RK4 step. accel_func can take (r) or (r, v).
    """
    import inspect
    n_args = len(inspect.signature(accel_func).parameters)

    def a(r, v):
        return accel_func(r, v) if n_args == 2 else accel_func(r)

    k1r, k1v = v,            a(r,            v)
    k2r, k2v = v + dt/2*k1v, a(r + dt/2*k1r, v + dt/2*k1v)
    k3r, k3v = v + dt/2*k2v, a(r + dt/2*k2r, v + dt/2*k2v)
    k4r, k4v = v + dt*k3v,   a(r + dt*k3r,   v + dt*k3v)

    r_new = r + dt/6 * (k1r + 2*k2r + 2*k3r + k4r)
    v_new = v + dt/6 * (k1v + 2*k2v + 2*k3v + k4v)
    return r_new, v_new


def _propagate_rk4(r0_km, v0_km_s, duration_hours, dt_seconds, accel_func):
    """
    Generic RK4 propagator for any acceleration model.
    Works in meters internally, returns positions in km.
    """
    r = r0_km * 1e3
    v = v0_km_s * 1e3

    positions = [r / 1e3]
    t, T = 0.0, duration_hours * 3600

    while t < T:
        r, v = _rk4_step(r, v, dt_seconds, accel_func)
        positions.append(r / 1e3)
        t += dt_seconds

    return np.array(positions)


# ── Public propagators ────────────────────────────────────────────────────────

def propagate_keplerian(r0_km, v0_km_s, duration_hours, dt_seconds=10.0):
    """RK4 with two-body gravity only — no perturbations."""
    return _propagate_rk4(r0_km, v0_km_s, duration_hours, dt_seconds,
                           _accel_keplerian)


def propagate_j2(r0_km, v0_km_s, duration_hours, dt_seconds=10.0):
    """RK4 with J2 oblateness perturbation."""
    return _propagate_rk4(r0_km, v0_km_s, duration_hours, dt_seconds,
                           _accel_j2)


def propagate_j2_drag(r0_km, v0_km_s, duration_hours, dt_seconds=10.0):
    """RK4 with J2 oblateness + atmospheric drag."""
    return _propagate_rk4(r0_km, v0_km_s, duration_hours, dt_seconds,
                           _accel_j2_drag)


# ── Ground track ──────────────────────────────────────────────────────────────

def get_ground_track(omm, t_start, duration_hours, dt_seconds=60.0):
    """
    Computes geographic ground track using Skyfield for accurate
    TEME -> lat/lon conversion.
    Returns lats, lons arrays.
    """
    from skyfield.api import EarthSatellite, load

    ts  = load.timescale()
    sat = EarthSatellite.from_omm(ts, omm)

    lats, lons = [], []
    t = t_start
    while t <= t_start + timedelta(hours=duration_hours):
        skyfield_t = ts.from_datetime(t)
        subpoint   = sat.at(skyfield_t).subpoint()
        lats.append(subpoint.latitude.degrees)
        lons.append(subpoint.longitude.degrees)
        t += timedelta(seconds=dt_seconds)

    return lats, lons