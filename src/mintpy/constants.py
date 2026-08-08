############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Feb 2022                           #
############################################################
# Recommend usage:
#   from mintpy.constants import SPEED_OF_LIGHT


# physical parameters
SPEED_OF_LIGHT = 299792458            # m/sec^2
GRAVITATIONAL_CONSTANT = 6.6743e-11   # m^3 / kg / sec^2, commonly denoted as G or "Big G"

# Earth radius
# equatorial radius: a = 6378.1370e3
# polar      radius: b = 6356.7523e3
# arithmetic mean radius: R_1 = (2 * a + b) / 3 = 6371.0088e3
#   defined by IUGG and used in geophysics
EARTH_RADIUS = 6371.0088e3   # the arithmetic mean radius in meters
EARTH_GRAVITATIONAL_PARAMETER = 3.986004418e14    # m^3 / sec^2, commonly denoted as μ = G * M


############################## Planetary Parameters ##############################
class PlanetaryBody():
    def __init__(self, name: str, radius: float, mass: float, surface_gravity: float):
        self.name = name
        self.radius = radius                    # m
        self.mass = mass                        # kg
        self.surface_gravity = surface_gravity  # m/sec^2

Mercury = PlanetaryBody(
    name = "Mercury",
    radius = 2440e3,
    mass = 3.3010e23,
    surface_gravity = 3.63,
)

Venus = PlanetaryBody(
    name="Venus",
    radius=6050e3,
    mass=4.86731e24,
    surface_gravity=8.83,
)

Earth = PlanetaryBody(
    name="Earth",
    radius=EARTH_RADIUS,
    mass=5.97220005e24,
    surface_gravity=9.80665,
)

Moon = PlanetaryBody(
    name="Moon",
    radius=1710e3,
    mass=7.348e22,
    surface_gravity=1.55,
)

Mars = PlanetaryBody(
    name="Mars",
    radius=3395e3,
    surface_gravity=3.92,
    mass=6.4273e23,
)

Jupiter = PlanetaryBody(
    name="Jupiter",
    radius=71500e3,
    mass=1.89852e27,
    surface_gravity=25.9,
)

Saturn = PlanetaryBody(
    name="Saturn",
    radius=60000e3,
    mass=5.6846e26,
    surface_gravity=11.38,
)
