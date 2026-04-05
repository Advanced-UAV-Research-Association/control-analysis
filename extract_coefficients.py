import re
import numpy as np
from pathlib import Path


def extract_inertia_tensor(filepath, use_tensor_sign_convention=True):
    """
    Extract the total inertia tensor from an OpenVSP MassProps text file.

    Parameters
    ----------
    filepath : str
        Path to the OpenVSP *_MassProps.txt file.
    use_tensor_sign_convention : bool
        If True, returns the standard rigid-body inertia tensor:
            [[ Ixx, -Ixy, -Ixz],
             [-Ixy,  Iyy, -Iyz],
             [-Ixz, -Iyz,  Izz]]
        If False, returns the raw values exactly as listed in the file:
            [[Ixx, Ixy, Ixz],
             [Ixy, Iyy, Iyz],
             [Ixz, Iyz, Izz]]

    Returns
    -------
    numpy.ndarray
        3x3 inertia tensor.
    dict
        Dictionary with scalar components.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # First try the summary block
    summary_pattern = re.compile(
        r'^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+'
        r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+'
        r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+Ixx,\s*Iyy,\s*Izz\s*$'
        r'.*?'
        r'^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+'
        r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+'
        r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+Ixy,\s*Ixz,\s*Iyz\s*$',
        re.MULTILINE | re.DOTALL
    )

    m = summary_pattern.search(text)

    if m:
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = map(float, m.groups())
    else:
        # Fallback: parse the Totals row
        totals_pattern = re.compile(
            r'^Totals\s+'
            r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+'   # Mass
            r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+'   # cgX
            r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+'   # cgY
            r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+'   # cgZ
            r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+'   # Ixx
            r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+'   # Iyy
            r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+'   # Izz
            r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+'   # Ixy
            r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+'   # Ixz
            r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',     # Iyz
            re.MULTILINE
        )

        m = totals_pattern.search(text)
        if not m:
            raise ValueError("Could not find inertia values in the OpenVSP MassProps file.")

        _, _, _, _, Ixx, Iyy, Izz, Ixy, Ixz, Iyz = map(float, m.groups())

    components = {
        "Ixx": Ixx,
        "Iyy": Iyy,
        "Izz": Izz,
        "Ixy": Ixy,
        "Ixz": Ixz,
        "Iyz": Iyz,
    }

    if use_tensor_sign_convention:
        tensor = np.array([
            [ Ixx, -Ixy, -Ixz],
            [-Ixy,  Iyy, -Iyz],
            [-Ixz, -Iyz,  Izz]
        ])
    else:
        tensor = np.array([
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz]
        ])

    return tensor, components


def extract_stab_coeff(filepath: str) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, float]]:
    """
    Parse an OpenVSP .stab stability file.

    Returns
    -------
    stability_derivatives : dict[str, float]
        Main aerodynamic stability derivatives using conventional notation.
        Examples:
            CL_alpha, CD_alpha, Cm_alpha,
            CY_beta, Cl_beta, Cn_beta,
            Cl_p, Cm_q, Cn_r, ...

    control_derivatives : dict[str, dict[str, float]]
        Control derivatives grouped by control function.
        For this file format:
            - ConGrp_1 -> elevator (symmetric elevon / pitch)
            - ConGrp_2 -> aileron (differential elevon / roll)

        Example:
            {
                "elevator": {"CL_delta": ..., "Cm_delta": ..., ...},
                "aileron": {"CY_delta": ..., "Cl_delta": ..., ...}
            }

    extras : dict[str, float]
        Useful additional quantities for analysis:
        reference geometry, CG, flight condition, trim/base coefficients,
        static margin, neutral point, etc.

    Notes
    -----
    - Assumes MKS-consistent values as exported from OpenVSP.
    - Alpha, beta, and control derivatives are per radian.
    - p, q, r derivatives are returned exactly as reported by OpenVSP.
    """

    text = Path(filepath).read_text()
    lines = text.splitlines()

    # ----------------------------
    # 1) Parse scalar metadata
    # ----------------------------
    raw_scalars: dict[str, float] = {}
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("*"):
            continue

        parts = s.split()
        if len(parts) >= 3 and parts[0].endswith("_"):
            try:
                raw_scalars[parts[0]] = float(parts[1])
            except ValueError:
                pass

    # ----------------------------
    # 2) Locate derivative table
    # ----------------------------
    table_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Coef"):
            table_start = i
            break

    if table_start is None:
        raise ValueError("Could not find derivative table starting with 'Coef'.")

    header = lines[table_start].split()

    derivative_rows: dict[str, dict[str, float]] = {}
    for line in lines[table_start + 1:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue

        parts = s.split()
        if len(parts) != len(header):
            if derivative_rows:
                break
            continue

        row_name = parts[0]
        try:
            values = [float(x) for x in parts[1:]]
        except ValueError:
            continue

        derivative_rows[row_name] = dict(zip(header[1:], values))

    if not derivative_rows:
        raise ValueError("No derivative rows were parsed from the derivative table.")

    # ----------------------------
    # 3) Parse result scalars
    # ----------------------------
    result_scalars: dict[str, float] = {}
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("*"):
            continue

        parts = s.split()
        if len(parts) >= 3:
            name = parts[0]
            if name in {"SM", "X_np"}:
                try:
                    result_scalars[name] = float(parts[1])
                except ValueError:
                    pass

    # ----------------------------
    # 4) Rename coefficients to conventional notation
    # ----------------------------
    coef_name_map = {
        "CL": "CL",     # lift
        "CD": "CD",     # drag
        "CS": "CY",     # side force
        "CMl": "Cl",    # rolling moment
        "CMm": "Cm",    # pitching moment
        "CMn": "Cn",    # yawing moment

        # Optional body-axis forms
        "CFx": "CX",
        "CFy": "CY_body",
        "CFz": "CZ",
        "CMx": "Cl_body",
        "CMy": "Cm_body",
        "CMz": "Cn_body",
    }

    major_rows = ["CL", "CD", "CS", "CMl", "CMm", "CMn"]

    stability_derivatives: dict[str, float] = {}

    derivative_suffixes = {
        "Alpha": "alpha",
        "Beta": "beta",
        "Mach": "Mach",
        "U": "u",
        "p": "p",
        "q": "q",
        "r": "r",
    }

    for row in major_rows:
        if row not in derivative_rows:
            continue

        conv_row = coef_name_map[row]
        row_data = derivative_rows[row]

        for col_name, suffix in derivative_suffixes.items():
            if col_name in row_data:
                stability_derivatives[f"{conv_row}_{suffix}"] = row_data[col_name]

    # ----------------------------
    # 5) Control derivatives
    # ----------------------------
    # This file format is fixed for your analyses:
    #   ConGrp_1 = pitch-symmetric elevon = elevator
    #   ConGrp_2 = roll-differential elevon = aileron
    control_column_map = {
        "ConGrp_1": "elevator",
        "ConGrp_2": "aileron",
    }

    control_derivatives: dict[str, dict[str, float]] = {
        "elevator": {},
        "aileron": {},
    }

    for ctrl_col, ctrl_name in control_column_map.items():
        for row in major_rows:
            if row in derivative_rows and ctrl_col in derivative_rows[row]:
                conv_row = coef_name_map[row]
                control_derivatives[ctrl_name][f"{conv_row}_delta"] = derivative_rows[row][ctrl_col]

    # ----------------------------
    # 6) Extras
    # ----------------------------
    extras = {
        # Reference geometry
        "S_ref": raw_scalars.get("Sref_"),
        "c_ref": raw_scalars.get("Cref_"),
        "b_ref": raw_scalars.get("Bref_"),

        # CG
        "x_cg": raw_scalars.get("Xcg_"),
        "y_cg": raw_scalars.get("Ycg_"),
        "z_cg": raw_scalars.get("Zcg_"),

        # Flight condition
        "Mach": raw_scalars.get("Mach_"),
        "alpha_deg": raw_scalars.get("AoA_"),
        "beta_deg": raw_scalars.get("Beta_"),
        "rho": raw_scalars.get("Rho_"),
        "V_inf": raw_scalars.get("Vinf_"),
        "roll_rate": raw_scalars.get("Roll__Rate"),
        "pitch_rate": raw_scalars.get("Pitch_Rate"),
        "yaw_rate": raw_scalars.get("Yaw___Rate"),

        # Base coefficients
        "CL_0": derivative_rows.get("CL", {}).get("Total"),
        "CD_0": derivative_rows.get("CD", {}).get("Total"),
        "CY_0": derivative_rows.get("CS", {}).get("Total"),
        "Cl_0": derivative_rows.get("CMl", {}).get("Total"),
        "Cm_0": derivative_rows.get("CMm", {}).get("Total"),
        "Cn_0": derivative_rows.get("CMn", {}).get("Total"),

        # Body-axis base coefficients
        "CX_0": derivative_rows.get("CFx", {}).get("Total"),
        "CY_body_0": derivative_rows.get("CFy", {}).get("Total"),
        "CZ_0": derivative_rows.get("CFz", {}).get("Total"),

        # Stability summary
        "static_margin": result_scalars.get("SM"),
        "x_neutral_point": result_scalars.get("X_np"),
    }

    extras = {k: v for k, v in extras.items() if v is not None}

    return stability_derivatives, control_derivatives, extras
