import re
import numpy as np

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

