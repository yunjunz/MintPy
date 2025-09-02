############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Aug 2025                           #
############################################################
# Recommend import:
#   from mintpy.geometry import orbit


import numpy as np


def get_relative_orbit_number(abs_orbit: int, platform: str) -> int:
    """Return the relative orbit number given the absolute orbit number.

    Parameters: abs_orbit - int, absolute orbit number
                platform  - str, SAR platform
    Returns:    rel_orbit - int, relative orbit number
    """
    if sensor == "S1A":
        rel_orbit = (abs_orbit - 73) % 175 + 1
    elif sensor == "S1B":
        rel_orbit = (abs_orbit - 27) % 175 + 1
    elif sensor == "S1C":
        rel_orbit = (abs_orbit - 172) % 175 + 1
    elif sensor == "TSX":
        rel_orbit = (abs_orbit - 1) % 167 + 1
    else:
        raise ValueError(f"Unknown platform: {platform}!")

    return rel_orbit


def get_los_azimuth_angle(orb_incl, orb_dir, sat_hgt, look_dir, lat, round_flag=False):
    """Get the LOS azimuth angle given the satellite parameters and ground latitude.

    Parameters: orb_incl     - float, orbit inclination angle in degree
                orb_dir      - str, orbit/pass direction, ascending or descending
                sat_hgt      - float, satellite height/altitude in meter
                look_dir     - str, radar/antenna look direction, left or right
                lat          - float, latitude of the point of interest on the ground
                round_flag   - bool, round the output angle to the nearest integer
    Returns:    los_az_angle - float, azimuth angle of the LOS vector from the ground to the SAR platform
                               measured from the north with anti-clockwise as positive in degrees
    """

    orb_az_angle = ground_track_azimuth(lat, orb_incl, sat_hgt, orb_dir)
    los_az_angle = orbit2los_azimuth_angle(orb_az_angle, look_dir)

    # convert to [0, 360)
    if np.isscalar(los_az_angle):
        los_az_angle = los_az_angle + 360 if los_az_angle < 0 else los_az_angle
    else:
        los_az_angle[los_az_angle<0] += 360

    # use the nearest integer angle for naming simplicity and re-use
    if round_flag:
        los_az_angle = np.round(los_az_angle)

    return los_az_angle


def orbit2los_azimuth_angle(orb_az_angle, look_direction='right'):
    """Convert the azimuth angle of the along-track vector to the LOS vector.
    Parameters: orb_az_angle - np.ndarray or float, azimuth angle of the SAR platform along track/orbit direction
                               measured from the north with anti-clockwise direction as positive, in the unit of degrees
    Returns:    los_az_angle - np.ndarray or float, azimuth angle of the LOS vector from the ground to the SAR platform
                               measured from the north with anti-clockwise direction as positive, in the unit of degrees
    """
    if look_direction == 'right':
        los_az_angle = orb_az_angle + 90
    else:
        los_az_angle = orb_az_angle - 90
    los_az_angle -= np.round(los_az_angle / 360.) * 360.
    return los_az_angle


def ground_track_azimuth(phi, orb_incl, h_sat, orb_dir="ascending"):
    """
    Compute ground-track heading H (bearing from North, anti-clockwise positive),
    including Earth's rotation, for a circular orbit.

    Parameters
    ----------
    phi : float / np.ndarray
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
    H_deg : float / np.ndarray
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

    # Convert to numpy array [to support array operation]
    phi_rad = np.array(phi_rad, dtype=np.float32)
    one = np.ones(phi_rad.size)

    # Compute sin u from latitude
    if abs(np.sin(orb_incl_rad)) < 1e-12:
        raise ValueError("Inclination too small; sin(orb_incl) ~ 0")
    sin_u = np.sin(phi_rad) / np.sin(orb_incl_rad)
    if np.any(np.abs(sin_u) > 1.0 + 1e-12):
        raise ValueError(f"No valid argument of latitude: |sin_u|={sin_u}")
    sin_u = np.maximum(-1*one, np.minimum(one, sin_u))  # clamp

    # Choose cos u sign from pass type
    cos_u_mag = np.sqrt(np.maximum(one*0.0, one*1.0 - sin_u*sin_u))
    if orb_dir.lower().startswith("a"):
        cos_u = cos_u_mag   # moving toward pole
    elif orb_dir.lower().startswith("d"):
        cos_u = cos_u_mag * -1   # moving away from pole
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
    H_rad[H_rad<0] += 2*np.pi
    H_deg = np.rad2deg(H_rad)

    # Heading: ensure with (-180, 180]
    H_deg[H_deg <= -180] += 360
    H_deg[H_deg > 180] -= 360

    # Output: ensure consistent data type as input
    if np.isscalar(phi):
        H_deg = H_deg[0]

    return H_deg
