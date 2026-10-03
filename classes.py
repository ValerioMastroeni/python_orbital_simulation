import numpy as np
from scipy.integrate import solve_ivp
import os
import pandas as pd

# Constants
G = 6.674e-11    # N*m^2 / Kg^2
AU = 149597870707      # m
MT = 5.97 * 10**24     # Kg
MS = 1.989 * 10**30    # Kg
YEAR = 60*60*24*365    # s
DAY = 60*60*24         # s

class CelestialObject:
    def __init__(self, name, mass: float, pos, vel=None, acc=None, F=None):
        self.name = name
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel if vel is not None else [0,0,0], dtype=float)
        self.acc = np.array(acc if acc is not None else [0,0,0], dtype=float)
        self.mass = mass
        self.F = F

    # Prints name, mass and coordinates in AUs
    def __repr__(self):
        unit = MS if self.mass >= 0.1 * MS else MT
        unit_str = "MS" if self.mass >= 0.1 * MS else "MT"
        return f"{self.name}({self.mass/unit:.4e} {unit_str}): [{self.pos[0]/AU:.3e} AU, {self.pos[1]/AU:.3e} AU, {self.pos[2]/AU:.3e} AU]"


def ResetPosition(obj):
    obj.pos = np.zeros(3)
    obj.vel = np.zeros(3)
    obj.acc = np.zeros(3)


# Orbital velocity for a circular orbit
def CircOrbitVel(pos):
    return np.sqrt((G * MS) / np.linalg.norm(pos))