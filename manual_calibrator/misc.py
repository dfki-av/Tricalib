__author__ = "Rahul Jakkamsetty"
__license__ = "MIT"
__doc__ = """
Few Miscellaneous functions. 
Developed at DFKI in JULY-AUGUST 2025.
"""

# python imports


# third-party imports
import numpy as np
from scipy.spatial.transform import Rotation as R

# internal imports


def normalize_quat(q: np.array) -> np.array:
    """
    Normalizes the provided quaternion (from scipy. [x, y, z, w])
    """
    return q/np.linalg.norm(q)


def quat_to_matrix(q: np.array) -> np.array:
    """
    Normalizes the quaternion and converts into a matrix.
    """

    q_normalized = normalize_quat(q)

    return R.from_quat(q_normalized).as_matrix()


def compose_T(R: np.array, t: np.array) -> np.array:
    """
    Composes the rotation matrix (3x3) and translation vector (3,) into a Transformation matrix (4x4)

    Args:
    -----------------
    R: 3x3 Rotation matrix
    t: translation vector
    returns: Transformation matrix (4x4)
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T


def decompose_T(T: np.array) -> tuple[np.array]:
    """
    Decomposes T(4x4) Transformation matrix into Rotation matrix(3x3) and translation vector(3,)

    """
    return T[:3, :3], T[:3, 3]
