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

def calc_vect(obj1, obj2):
    return obj2.pos - obj1.pos

def grav_force(obj1, obj2):
    r_v = calc_vect(obj1, obj2)
    r = np.linalg.norm(r_v)
    if np.isclose(r, 0):
        return np.zeros(3,dtype=float)
    else:
        return (G * obj1.mass * obj2.mass) / (r**2) * (r_v / r)

# Calculates position, velocity and acceleration for an object, given the forces that act on it
# F = sum of the forces, dT update time
# Only works for a single moving objects, and all the others are fixed
def VelocityVerlet(obj1, others, dT):

    new_pos = obj1.pos + obj1.vel * dT + obj1.acc * ((dT**2) * 0.5)

    F_new = sum(grav_force(obj1, obj) for obj in others if obj is not obj1)

    new_acc = F_new / obj1.mass

    new_vel = obj1.vel + (obj1.acc + new_acc) * (dT * 0.5)

    obj1.pos = new_pos
    obj1.vel = new_vel
    obj1.acc = new_acc

# Velocity-Verlet for N objects
def N_VelocityVerlet(objs, dT):
    
    for obj in objs:
        obj.pos = obj.pos + obj.vel * dT + obj.acc * ((dT**2) * 0.5)

    old_accs = [obj.acc.copy() for obj in objs]

    for obj in objs:
        obj.F = np.zeros(3)

    # Calculates half of the forces, since F(2->1) = -F(1->2)
    N = len(objs)
    for i in range(N):
        for j in range(i + 1, N):
            F = grav_force(objs[i], objs[j])
            objs[i].F += F
            objs[j].F -= F
    
    for obj in objs:
        obj.acc = obj.F / obj.mass

    for obj, old_acc in zip(objs, old_accs):
        obj.vel = obj.vel + (old_acc + obj.acc) * (dT * 0.5)

def _calc_acc(pos, masses):
    N = len(masses)
    acc = np.zeros((N,3))
    for i in range(N):
        for j in range(N):
            if i != j:
                r_vec = pos[j] - pos[i]
                r = np.linalg.norm(r_vec)
                acc[i] += G * masses[j] * r_vec / r**3
    return acc

def _n_body_der(t, y, masses):
    """
    y = 1D array, y = [x0, y0, ..., zN, vx0, vy0, ..., vzN]
    """
    N = len(masses)
    pos = y[:3*N].reshape((N, 3))
    vel = y[3*N:].reshape((N, 3))

    acc = _calc_acc(pos, masses)
    dydt = np.zeros_like(y)
    dydt[:3*N] = vel.flatten()
    dydt[3*N:] = acc.flatten()

    return dydt

def RK45(objs, tSpan, step, rtol = 1e-9, atol = 1e-12):
    
    """

    objs = array of CelestialObjects
    tSpan = [0, end (in seconds)]

    returns pos, vel: [N, 3 (x,y,z), (tSpan/step)+1] (includes the starting values)
    objs[i].pos, objs[i].vel = pos[i, :, k], vel[i, :, k]
    con k istante di tempo

    pos[i, :, 0] = initial pos, pos[i, :, -1] = final pos

    """

    pos = [np.array(obj.pos, dtype=float) for obj in objs]
    vel = [np.array(obj.vel, dtype=float) for obj in objs]
    y0 = np.concatenate([np.array(pos).flatten(), np.array(vel).flatten()])
    masses = [np.array(obj.mass, dtype=float) for obj in objs]

    t_eval = np.linspace(0, tSpan[1], int(tSpan[1]/step)+1)
    sol = solve_ivp(_n_body_der, tSpan, y0, t_eval=t_eval, rtol=rtol, atol=atol, args=(masses,))

    N = len(masses)
    num_t = len(sol.t)

    pos = sol.y[:3*N, :].reshape(N, 3, num_t)
    vel = sol.y[3*N:, :].reshape(N, 3, num_t)

    return pos, vel

def DOP853(objs, tSpan, step, rtol = 1e-9, atol = 1e-12):
    
    """

    objs = array of CelestialObjects
    tSpan = [0, end (in seconds)]

    returns pos, vel: [N, 3 (x,y,z), (tSpan/step)+1] (includes the starting values)
    objs[i].pos, objs[i].vel = pos[i, :, k], vel[i, :, k]
    con k istante di tempo

    pos[i, :, 0] = initial pos, pos[i, :, -1] = final pos

    """

    pos = [np.array(obj.pos, dtype=float) for obj in objs]
    vel = [np.array(obj.vel, dtype=float) for obj in objs]
    y0 = np.concatenate([np.array(pos).flatten(), np.array(vel).flatten()])
    masses = [np.array(obj.mass, dtype=float) for obj in objs]

    t_eval = np.linspace(0, tSpan[1], int(tSpan[1]/step)+1)
    sol = solve_ivp(_n_body_der, tSpan, y0, method="DOP853", t_eval=t_eval, rtol=rtol, atol=atol, args=(masses,))

    N = len(masses)
    num_t = len(sol.t)

    pos = sol.y[:3*N, :].reshape(N, 3, num_t)
    vel = sol.y[3*N:, :].reshape(N, 3, num_t)

    return pos, vel