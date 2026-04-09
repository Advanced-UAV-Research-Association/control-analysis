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
# We know D is 0

# %%
D_pitch = np.zeros((3, 1))
D_roll= np.zeros((2, 1))
