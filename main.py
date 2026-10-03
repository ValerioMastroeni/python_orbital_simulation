from API_functions import *
from classes import *
from integrators import *
import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

plt.style.use('dark_background')

# objects creation and constant definition

years = 20
start = 2025

bis_years = 0
if years == start % 4:
    bis_years += 1
elif years > start % 4:
    bis_years += int(years / 4)

START = f"{start}-01-01"
STOP = f"{2025+years}-01-01"
STEP = "1d"

objects_id = ["10","199","299","301","399","499","599","699","799","899"]
objects_name = ["Sun","Mercury","Venus","Moon","Earth","Mars","Jupiter","Saturn","Uranus","Neptune"]
objects_mass = [1.98847e30, 3.3011e23, 4.8675e24, 7.342e22, 5.97237e24,
                6.4171e23, 1.8982e27, 5.6834e26, 8.6810e25, 1.02413e26]

acc_data = [Horizon_get_data_cached(id, START, STOP, STEP) for id in objects_id]

objectsVV = [
    CelestialObject(name, mass,
                    np.array(data.iloc[0, 2:5], dtype=float) * 1000,
                    np.array(data.iloc[0, 5:8], dtype=float) * 1000)
    for name, mass, data in zip(objects_name, objects_mass, acc_data)
]

objectsRK = [
    CelestialObject(name, mass,
                    np.array(data.iloc[0, 2:5], dtype=float) * 1000,
                    np.array(data.iloc[0, 5:8], dtype=float) * 1000)
    for name, mass, data in zip(objects_name, objects_mass, acc_data)
]

N = len(objectsVV)
# array (Steps * N * 3); obj_pos[time_tick, object_number, dimension(x,y,z)]
obj_pos_VV = [objectsVV[i].pos for i in range(N)]
obj_pos_RK = [objectsRK[i].pos for i in range(N)]

# simulation setup

dt = 3600
nsteps = int(((years*365*DAY)+bis_years)/dt)

# simulation
# VV
obj_pos_VV = [np.array([obj.pos for obj in objectsVV], dtype=float)]
for _ in range(nsteps):
    N_VelocityVerlet(objectsVV, dt)
    obj_pos_VV.append(np.array([obj.pos for obj in objectsVV], dtype=float))

# RK45
obj_pos_RK, vel = RK45(objectsRK, [0, int(nsteps*dt)], dt)

# transpose puts axis in the indicated order, so here swaps the 1st and 2nd, to have VV in the same format as acc_data
# VV = [time, N, (x,y,z)] -> [N, time, (x,y,z)]
# RK = [N, (x,y,z), time] -> [N, time, (x,y,z)]
VV = np.array(obj_pos_VV, dtype=float).transpose(1, 0, 2)
RK = np.array(obj_pos_RK, dtype=float).transpose(0, 2, 1)

# plot

# delta |r| for each object
Horizon_data = np.stack([np.array(data.iloc[0:(years*365)+1, 2:5], dtype=float) for data in acc_data]) # array [N, time, (x,y,z)]
VV_daily = VV[:, ::int(DAY/dt), :]
RK_daily = RK[:, ::int(DAY/dt), :]

deltaE_VV = VV_daily - Horizon_data*1000
deltaE_RK = RK_daily - Horizon_data*1000

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# first object to not be displayed (max=N)(5 -> only first 4 objects)
M = N

# orbits
for i in range(M): # VV
    if i == 0:
        axs[0, 0].plot(VV[i, :, 0]/AU, VV[i, :, 1]/AU, marker="o", label=objects_name[i])
    else:
        axs[0, 0].plot(VV[i, :, 0]/AU, VV[i, :, 1]/AU, label=objects_name[i])

for i in range(M): # RK
    if i == 0:
        axs[0, 1].plot(RK[i, :, 0]/AU, RK[i, :, 1]/AU, marker="o", label=objects_name[i])
    else:
        axs[0, 1].plot(RK[i, :, 0]/AU, RK[i, :, 1]/AU, label=objects_name[i])

# error
for i in range(M):
    axs[1, 0].plot(range(years*365+1), np.linalg.norm(deltaE_VV[i], axis=1), label=objects_name[i])
    axs[1, 1].plot(range(years*365+1), np.linalg.norm(deltaE_RK[i], axis=1), label=objects_name[i])

axs[0, 0].axis("equal")
axs[0, 0].legend()
axs[0, 0].set_title("Orbits, VV")
axs[0, 1].axis("equal")
axs[0, 1].legend()
axs[0, 1].set_title("Orbits, RK")
axs[1, 0].set_ylim(bottom=0)
axs[1, 0].legend()
axs[1, 0].set_title("Error, VV")
axs[1, 1].set_ylim(bottom=0)
axs[1, 1].legend()
axs[1, 1].set_title("Error, RK")

plt.tight_layout()
plt.show()