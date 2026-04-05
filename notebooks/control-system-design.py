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

import sys
from pathlib import Path

# Add parent directory to path to find extract_coefficients
sys.path.insert(0, str(Path.cwd().parent))

import extract_coefficients
