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
print(inertia_tensor)

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
