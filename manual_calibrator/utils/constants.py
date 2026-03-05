__author__ = "Rahul Jakkamsetty"
__license__ = "CC BY-NC-SA 4.0"
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


"""
🔑 Coordinate System Conventions

LiDARs and cameras typically use different axis conventions:

- Camera pinhole model (OpenCV / vision):
  +X → right
  +Y → down
  +Z → forward (into the scene)

- LiDAR / robotics (ROS, ENU):
  +X → forward
  +Y → left
  +Z → up

When projecting LiDAR points to the RGB camera coordinate system
using the extrinsics [R|t], the origins and orientations are aligned,
but the axis conventions may still differ.

The fixed matrix M is a static rotation matrix (change of basis) that
converts from the LiDAR/robotics convention into the computer vision
convention expected by the camera projection model.

- Extrinsics: where the camera is relative to the LiDAR
- BASIS_MATRIX: how their axes are defined
"""


BASIS_MATRIX = np.array([[0, -1, 0],
                         [0, 0, -1],
                         [1, 0, 0]])


DSEC_R_RECT_EVENT = np.array([[0.99986606, -0.00319364,  0.01605171],
                              [0.00322964,  0.99999233, -0.00221712],
                              [-0.01604451,  0.00226867,  0.9998687],
                              ])

DSEC_R_RECT_RGB = np.array([[0.99988586, -0.01351071, -0.00676206 ],
                            [0.01353521,  0.99990195,  0.0035898],
                            [0.0067129, -0.00368091,  0.99997069],
                            ])
DSEC_T_GT = np.array([[0.99973298,  0.00994674,  0.02085725, -0.04372224],
                      [-0.01003579,  0.99994095,  0.0041691,  0.00101557],
                      [-0.02081454, -0.0043773,  0.99977377, -0.01337267],
                      [0.,  0.,  0.,  1.]])
