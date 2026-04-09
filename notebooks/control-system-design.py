# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Control System Design
# Here we will design the control system for the plane, we will use LQR to find the optimal poles

# %%
# %load_ext autoreload
# %autoreload 2

import control as ct
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

import os
import sys
from pathlib import Path
# Add parent directory to path to find extract_coefficients
sys.path.insert(0, str(Path.cwd().parent))

import extract_coefficients

# %% [markdown]
# ### Define paths and output coefficents

# %%
# define paths
Mass_Analysis_File = os.path.abspath("../data/flying_wing_better1_MassProps.txt")
Stab_Analysis_File = os.path.abspath("../data/flying_wing_better1.stab")

# %%
# test getting inertial tensor
inertia_tensor, components = extract_coefficients.extract_inertia_tensor(Mass_Analysis_File)
print(f"Ineria Tensor:\n{inertia_tensor}")
print("\nComponents:")
for k, v in components.items():
    print(f"{k:16s} = {v}")

# %%
# test getting stability coefficents
stability, controls, extras = extract_coefficients.extract_stab_coeff(Stab_Analysis_File)

print("Stability derivatives:")
for k, v in stability.items():
    print(f"{k:12s} = {v: .6f}")

print("\nControl derivatives:")
for surface, vals in controls.items():
    print(surface)
    for k, v in vals.items():
        print(f"  {k:10s} = {v: .6f}")

print("\nExtras:")
for k, v in extras.items():
    print(f"{k:16s} = {v}")	

# %% [markdown]
# ## Constants and Coefficents
# Here we need to define some constants, such as air density, etc.
# We need to linearize around a specific speed too, let's say ~30mph.

# %%
# defined constants
# V_0 = 13.4 # m/s ~= 30mph
V_0 = 18.49 # m/s the secretary said this was our trim speed for some reason
mass = 0.61373

rho = 1.225

# %% [markdown]
# ## calculate coefficients for the state space system
# ### Pitch Axis
#
# State vector:
# $$\mathbf{x}_{lon} = \begin{bmatrix} \alpha \\ q \\ \theta \end{bmatrix}$$
#
# $$A_{lon} = \begin{bmatrix} Z_\alpha & 1 & 0 \\ M_\alpha & M_q & 0 \\ 0 & 1 & 0 \end{bmatrix}, \quad
# B_{lon} = \begin{bmatrix} Z_{\delta_e} \\ M_{\delta_e} \\ 0 \end{bmatrix}$$
#
# <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
# <div>
#
# $$Z_\alpha = -\frac{q_\infty S}{m V_0} C_{L_\alpha}$$
#
# $$M_\alpha = \frac{q_\infty S \bar{c}}{I_{yy}} C_{m_\alpha}$$
#
# $$M_q = \frac{q_\infty S \bar{c}^2}{2 V_0 I_{yy}} C_{m_q}$$
#
# </div>
# <div>
#
# $$Z_{\delta_e} = -\frac{q_\infty S}{m V_0} C_{L_{\delta_e}}$$
#
# $$M_{\delta_e} = \frac{q_\infty S \bar{c}}{I_{yy}} C_{m_{\delta_e}}$$
#
# </div>
# </div>
#
# ---
#
# ### Roll Axis
#
# State vector:
# $$\mathbf{x}_{roll} = \begin{bmatrix} p \\ \phi \end{bmatrix}$$
#
# $$A_{roll} = \begin{bmatrix} L_p & 0 \\ 1 & 0 \end{bmatrix}, \quad
# B_{roll} = \begin{bmatrix} L_{\delta_a} \\ 0 \end{bmatrix}$$
#
# <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
# <div>
#
# $$L_p = \frac{q_\infty S b^2}{2 V_0 I_{xx}} C_{l_p}$$
#
# </div>
# <div>
#
# $$L_{\delta_a} = \frac{q_\infty S b}{I_{xx}} C_{l_{\delta_a}}$$
#
# </div>
# </div>

# %%
# get system coefficients

q_inf = 0.5 * rho * V_0**2
MAC = extras['c_ref']
b_ref = extras['b_ref']
S_ref = extras['S_ref']

Ixx = components['Ixx']
Iyy = components['Iyy']

# calculate matrix coefficients
# Pitch Axis (Longitudinal)
Z_alpha = -q_inf * S_ref / (mass * V_0) * stability['CL_alpha']
M_alpha = q_inf * S_ref * MAC / Iyy * stability['Cm_alpha']
M_q = q_inf * S_ref * MAC**2 / (2 * V_0 * Iyy) * stability['Cm_q']
Z_delta_e = -q_inf * S_ref / (mass * V_0) * controls['elevator']['CL_delta']
M_delta_e = q_inf * S_ref * MAC / Iyy * controls['elevator']['Cm_delta']

# Roll Axis
L_p = q_inf * S_ref * b_ref**2 / (2 * V_0 * Ixx) * stability['Cl_p']
L_delta_a = q_inf * S_ref * b_ref / Ixx * controls['aileron']['Cl_delta']

# %% [markdown]
# ## Construct State Matrices

# %%
# Construct A matrices
A_pitch = np.array([
    [Z_alpha, 1, 0],
    [M_alpha, M_q, 0],
    [0, 1, 0]
])

A_roll = np.array([
    [L_p, 0],
    [1, 0]
])

# Construct B matrices
B_pitch = np.array([
    [Z_delta_e],
    [M_delta_e],
    [0]
])

B_roll = np.array([
    [L_delta_a],
    [0]
])

# define the C matrices for the control library
C_pitch = np.eye(3)
C_roll= np.eye(2)

# D vectors, both zero
D_pitch = np.zeros((3, 1))
D_roll= np.zeros((2, 1))

# %% [markdown]
# ## Construct system and perform analysis

# %%
sys_pitch = ct.ss(A_pitch, B_pitch, C_pitch, D_pitch)
sys_roll = ct.ss(A_roll, B_roll, C_roll, D_roll)

# Check controllability
controllability_pitch = ct.ctrb(sys_pitch.A, sys_pitch.B)
controllability_roll = ct.ctrb(sys_roll.A, sys_roll.B)

rank_pitch = np.linalg.matrix_rank(controllability_pitch)
rank_roll = np.linalg.matrix_rank(controllability_roll)

print(f"Pitch system controllability matrix rank: {rank_pitch} (expected {sys_pitch.A.shape[0]})")
print(f"Roll system controllability matrix rank: {rank_roll} (expected {sys_roll.A.shape[0]})")

if rank_pitch == sys_pitch.A.shape[0]:
    print("Pitch system is CONTROLLABLE")
else:
    print("Pitch system is NOT controllable")

if rank_roll == sys_roll.A.shape[0]:
    print("Roll system is CONTROLLABLE")
else:
    print("Roll system is NOT controllable")

# %% [markdown]
# ## Open Loop Response Simulation
# Simulate the open loop response of each system with initial state perturbations

# %%
# Define initial states for each system
# Pitch system: small perturbation in angle of attack
x0_pitch = np.array([0.1, 0.0, 0.1])  # [alpha, q, theta]

# Roll system: small perturbation in roll rate
x0_roll = np.array([0.1, 0.0])  # [p, phi]

# Simulation time
T = 5  # seconds
t = np.linspace(0, T, 1000)

# Simulate open loop response
response_pitch = ct.initial_response(sys_pitch, t, X0=x0_pitch)
response_roll = ct.initial_response(sys_roll, t, X0=x0_roll)

# %% [markdown]
# ### Pitch Axis Response

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

states_pitch = ['Angle of Attack (rad)', 'Pitch Rate (rad/s)', 'Pitch Angle (rad)']

for i, (ax, state) in enumerate(zip(axes, states_pitch)):
    ax.plot(response_pitch.time, response_pitch.y[i], 'b-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(state)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Pitch Axis - {state}')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Roll Axis Response

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

states_roll = ['Roll Rate (rad/s)', 'Roll Angle (rad)']

for i, (ax, state) in enumerate(zip(axes, states_roll)):
    ax.plot(response_roll.time, response_roll.y[i], 'r-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(state)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Roll Axis - {state}')

plt.tight_layout()
plt.show()

# %%
