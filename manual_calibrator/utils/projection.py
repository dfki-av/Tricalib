__author__ = "Rahul Jakkamsetty"
__license__ = "MIT"
__doc__ = """
Few projection related functions. 
Developed at DFKI in DEC-JAN 2024-25.
"""
# python imports

from typing import Optional
# third-party imports
import numpy as np
import cv2

# internal imports
from btc_calib.utils.constants import CAMERA4_C2_DISTORTION, CAMERA4_C2_KMATRIX, UNIFICATION_MATRIX


def undistort_fisheye(image, return_newk=False):
    """
    undistorts the image assuming fisheye lens.
    Parameters:
    ----------
    image: np.array
    returns: undistored image (np.array)
    """
    dist = np.array([CAMERA4_C2_DISTORTION['k1'],
                     CAMERA4_C2_DISTORTION['k2'],
                     CAMERA4_C2_DISTORTION['k3'],
                     CAMERA4_C2_DISTORTION['k4']])

    img_shape = np.array(image.shape[:2])[::-1]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(CAMERA4_C2_KMATRIX,
                                                                   dist, img_shape, np.eye(3))
    undistort_image = cv2.fisheye.undistortImage(
        image, CAMERA4_C2_KMATRIX,
        dist, Knew=new_K
    )
    if return_newk:
        return undistort_image, new_K

    return undistort_image


def project_points(points_3d: np.array,
                   rotation_matrix: np.array,
                   translation_vector: np.array,
                   camera_matrix: np.array, unification=False) -> np.array:
    """
    Projects the 3D points in LiDAR to image plane.

    points_3d: point in lidar co-ordinate system (Nx3)
    rotation_matrix: orienation transformation (3x3)
    translation_vector: 3x1 vector corresponding to x, y, z.
    camera_matrix: 3x3 camera intrinsic matrix.

    returns: projected 2d points in image plane.
    """

    # Apply the rotation and translation to the 3D points
    transformed_points = np.dot(
        rotation_matrix, points_3d.T).T + translation_vector

    # If required project to original camera co-ordinate system.
    if unification:
        transformed_points = np.dot(UNIFICATION_MATRIX, transformed_points.T).T
    # Project points to 2D using the camera intrinsic matrix
    points_2d = np.dot(camera_matrix, transformed_points.T).T

    # Normalize by the depth (z coordinate)
    points_2d[:, 0] /= points_2d[:, 2]
    points_2d[:, 1] /= points_2d[:, 2]
    points_2d = points_2d[:, :2]  # Extract only x and y

    return points_2d


def visualize_projection(image: np.array, points_3d: np.array,
                         points_2d: np.array, intensities: Optional[np.array]  = None,
                         color_map: int = cv2.COLORMAP_JET) -> np.array:
    """
    visualizes the projection of point cloud using depth or intensity information.
    basically points are colored using this information.

    Parameters:
    -----------
    image: an image array on which the points are painted.
    points_3d: lidar point cloud (not transformed).
    points_2d: projected point cloud in 2D plane.
    intensities: array of intensity values. when None provided, defaults to depth values.
    color_map: cv2 color map used to paint the projected point cloud.

    returns: point cloud painted image.
    """

    valid_mask = points_3d[:, 0] > 0
    points_3d = points_3d[valid_mask]
    points_2d = points_2d[valid_mask]
    if intensities is None:
        painting_values = points_3d[:, 0]  # x-coordinate represents depth
    else:
        painting_values = intensities[valid_mask]
    # Normalize the depth values to map them to colors
    paint_normalized = cv2.normalize(
        painting_values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colors = cv2.applyColorMap(paint_normalized, color_map)
    for i, point in enumerate(points_2d):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            color = tuple(int(c) for c in colors[i].flatten().tolist())
            cv2.circle(image, (x, y), 3, color, -1)
    return image


def normalize_pixels(points, K):
    """
    pixel coordinates --> normalized coordinates using Intrinsic matrix.
    """
    pts_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    pts_norm = (np.linalg.inv(K)@pts_hom.T).T
    return pts_norm[:, :2]