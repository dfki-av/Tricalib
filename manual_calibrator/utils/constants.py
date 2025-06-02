__author__ = "Rahul Jakkamsetty"
__license__ = "MIT"
__doc__ = """
Few camera related constants.
Developed at DFKI in DEC-JAN 2024-25.
"""

# python imports


# third-party imports
import numpy as np

# internal imports

CAMERA4_C2_KMATRIX = np.array([
    [1943.294701380089, 0.0, 962.1996122776545],
    [0.0, 1946.4071649380578, 553.5785864000259],
    [0.0,   0.0,    1.0]])

CAMERA4_C2_DISTORTION = dict(
    k1=-0.22257277754764532,
    k2=-0.2466768448392368,
    k3=0.7975777282988225,
    k4=-1.0639627638034164)

UNIFICATION_MATRIX = np.array([[0, -1, 0],
                               [0, 0, -1],
                               [1, 0, 0]])
