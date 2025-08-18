############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Aug 2025                           #
############################################################
# Recommend import:
#   from mintpy.geometry import orbit


import numpy as np


def ground_track_azimuth(phi, orb_incl, h_sat, orb_dir="ascending"):
    """
    Compute ground-track heading H (bearing from North, anti-clockwise positive),
    including Earth's rotation, for a circular orbit.

    Parameters
    ----------
    phi : float
        Geodetic latitude of subsatellite point (deg, positive north).
    orb_incl : float
        Orbital inclination (deg, 0..180).
    h_sat : float
        Satellite altitude above mean Earth radius (m).
    orb_dir : str, optional
        'ascending'  -> northbound (lat increasing, moving toward pole)
        'descending' -> southbound (lat decreasing, moving away from pole)

    Returns
    -------
    H_deg : float
        Ground-track heading in degrees, -180..180,
        with 0 = North, anti-clockwise positive.
    """

    # Constants
    mu = 3.986004418e14     # m^3/s^2, Earth's GM (standard gravitational parameter)
    Omega_e = 7.2921150e-5  # rad/s, Earth rotation rate (sidereal)
    R_e = 6.371e6           # m, mean Earth radius

    # Convert to radians
    phi_rad = np.deg2rad(phi)
    orb_incl_rad = np.deg2rad(orb_incl)

    # Compute sin u from latitude
    if abs(np.sin(orb_incl_rad)) < 1e-12:
        raise ValueError("Inclination too small; sin(orb_incl) ~ 0")
    sin_u = np.sin(phi_rad) / np.sin(orb_incl_rad)
    if abs(sin_u) > 1.0 + 1e-12:
        raise ValueError(f"No valid argument of latitude: |sin_u|={sin_u}")
    sin_u = max(-1.0, min(1.0, sin_u))  # clamp

    # Choose cos u sign from pass type
    cos_u_mag = np.sqrt(max(0.0, 1.0 - sin_u*sin_u))
    if orb_dir.lower().startswith("a"):
        cos_u = +cos_u_mag   # moving toward pole
    elif orb_dir.lower().startswith("d"):
        cos_u = -cos_u_mag   # moving away from pole
    else:
        raise ValueError("orb_dir must be 'ascending' or 'descending'")

    # Satellite mean motion
    r = R_e + h_sat
    Omega_s = np.sqrt(mu / (r**3))

    # Velocity components relative to Earth
    V_N = R_e * (Omega_s * np.sin(orb_incl_rad) * cos_u) / np.cos(phi_rad)
    V_E = R_e * (Omega_s * np.cos(orb_incl_rad) - Omega_e * np.cos(phi_rad))

    # Heading (North = 0, anticlockwise positive)
    H_rad = np.arctan2(-V_E, V_N)
    if H_rad < 0:
        H_rad += 2*np.pi
    H_deg = np.rad2deg(H_rad)

    # Heading: ensure with (-180, 180]
    if H_deg <= -180:
        H_deg += 360
    elif H_deg > 180:
        H_deg -= 360
    return H_deg