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
from PyQt6.QtGui import QPixmap, QImage

# internal imports


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """
    Normalizes the provided quaternion (from scipy. [x, y, z, w])
    """
    return q/np.linalg.norm(q)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """
    Normalizes the quaternion and converts into a matrix.
    """

    q_normalized = normalize_quat(q)

    return R.from_quat(q_normalized).as_matrix()


def compose_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
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


def decompose_T(T: np.ndarray) -> tuple[np.ndarray]:
    """
    Decomposes T(4x4) Transformation matrix into Rotation matrix(3x3) and translation vector(3,)

    """
    return T[:3, :3], T[:3, 3]


def image_to_pixmap(img: np.ndarray):
    h, w, ch = img.shape
    bytes_per_line = ch * w
    q_image = QImage(img.data, w, h, bytes_per_line,
                     QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(q_image)
    return pixmap


def geodesic_distance_from_rotm(R1: np.ndarray, R2: np.ndarray, fix_numeric: bool = False):
    """
    Returns (theta, axis) where theta is the geodesic distance in degrees
    between orientations R1 and R2, and axis is the rotation axis (unit vector).
    If axis is None, angle is 0 (no rotation) or ambiguous.
    """

    R = R2 @ R1.T            # relative rotation R_rel

    if fix_numeric:
        # project to nearest orthogonal matrix with det=+1 (SVD fix)
        U, S, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            # fix improper rotation
            U[:, -1] *= -1
            R = U @ Vt

    tr = np.trace(R)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # axis extraction
    if np.isclose(theta, 0.0):
        axis = None  # no unique axis
    elif np.isclose(theta, np.pi):
        # 180 degrees: extract axis from diagonal (numerically careful)
        axis = np.array([
            np.sqrt(max(0.0, (R[0, 0] + 1.0) / 2.0)),
            np.sqrt(max(0.0, (R[1, 1] + 1.0) / 2.0)),
            np.sqrt(max(0.0, (R[2, 2] + 1.0) / 2.0))
        ])
        # fix signs using off-diagonals
        axis[0] = np.copysign(axis[0], R[2, 1] - R[1, 2])
        axis[1] = np.copysign(axis[1], R[0, 2] - R[2, 0])
        axis[2] = np.copysign(axis[2], R[1, 0] - R[0, 1])
        if np.linalg.norm(axis) < 1e-8:
            axis = None
        else:
            axis = axis / np.linalg.norm(axis)
    else:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] -
                        R[2, 0], R[1, 0] - R[0, 1]])
        axis = axis / (2.0 * np.sin(theta))

    return np.degrees(theta), axis
