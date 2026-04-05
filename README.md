# UAV State Space Control System Design

This repository contains tools and notebooks for designing a state space control system for a UAV using Linear Quadratic Regulator (LQR) techniques.

## Project Structure

- **[`data/`](data/)** - Contains pre-simulated coefficient files from OpenVSP analysis:
  - `flying_wing_better1_MassProps.txt` - Mass properties and inertia tensor data
  - `flying_wing_better1.stab` - Stability and control derivative coefficients

- **[`notebooks/`](notebooks/)** - Jupyter notebooks (tracked as Python files via Jupytext):
  - [`control-system-design.py`](notebooks/control-system-design.py) - Main notebook for designing the LQR control system
  - [`jupytext.toml`](notebooks/jupytext.toml) - Configuration for pairing notebooks with Python files

- **[`extract_coefficients.py`](extract_coefficients.py)** - Python module for extracting stability derivatives and inertia tensors from OpenVSP output files

- **[`.devcontainer/`](.devcontainer/)** - Development container configuration for reproducible environments

## Development Setup

This project uses a devcontainer with a pre-configured Python environment. The container includes:
- Python 3.12
- Required dependencies from [`requirements.txt`](requirements.txt)

### Key Dependencies
- `slycot` - Python control systems library (requires Fortran compiler)
- `control` - Python control systems library
- `scipy`, `numpy`, `pandas`, `matplotlib` - Scientific computing stack
- `jupytext` - Tool for pairing Jupyter notebooks with Python scripts

## Jupytext Workflow

Instead of tracking Jupyter notebooks (`.ipynb`) in git, this project uses Jupytext to pair notebooks with Python files (`.py`). The `ipynb` files are gitignored, while the `py:percent` format Python files are tracked. This provides:
- Better diff visibility in version control
- Compatibility with standard Python tooling
- Same functionality as full notebooks

To work with the notebooks, open the `.py` files in VS Code with Jupyter support or convert them back to notebook format using `jupytext`.
