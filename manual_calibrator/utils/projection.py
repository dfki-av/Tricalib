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
from manual_calibrator.utils.constants import (CAMERA4_C2_DISTORTION, CAMERA4_C2_KMATRIX, BASIS_MATRIX,
                                               DSEC_R_RECT_EVENT, DSEC_R_RECT_RGB, DSEC_T_GT)


def undistort_fisheye(image, return_newk=False):
    """
    undistorts the image assuming fisheye lens.
    Parameters:
    ----------
    image: np.ndarray
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


def project_points(points_3d: np.ndarray,
                   rotation_matrix: np.ndarray,
                   translation_vector: np.ndarray,
                   camera_matrix: np.ndarray, unification: bool = False,
                   rectification_matrix: np.ndarray | None = None) -> np.ndarray:
    """
    Projects the 3D points in LiDAR to image plane.

    points_3d: point in lidar co-ordinate system (Nx3)
    rotation_matrix: orienation transformation (3x3)
    translation_vector: 3x1 vector corresponding to x, y, z.
    camera_matrix: 3x3 camera intrinsic matrix.

    returns: projected 2d points in image plane.
    """

    # Apply the rotation and translation to the 3D points

    if unification:
        rotation_matrix = rotation_matrix@BASIS_MATRIX

    if rectification_matrix is None:
        rectification_matrix = np.eye(3)

    rotation_matrix = rotation_matrix@rectification_matrix

    transformed_points = np.dot(
        rotation_matrix, points_3d.T).T + translation_vector
    # If required project to original camera co-ordinate system.

    # Project points to 2D using the camera intrinsic matrix
    points_2d = np.dot(camera_matrix, transformed_points.T).T

    # Normalize by the depth (z coordinate)
    points_2d[:, 0] /= points_2d[:, 2]
    points_2d[:, 1] /= points_2d[:, 2]
    points_2d = points_2d[:, :2]  # Extract only x and y

    return points_2d


def project_rgb_to_event(points_rgb, K_rgb, K_ev, extrinsics, rect_matrices):
    """
    Projects 2D points from the RGB image to the event image.

    Parameters:
    -----------
    points_rgb: (N, 2) array of 2D points in the RGB image.
    K_rgb: (3, 3) intrinsic matrix of the RGB camera.
    K_ev: (3, 3) intrinsic matrix of the event camera.
    extrinsics: (4, 4) transformation matrix from event to RGB.

    Returns:
    --------
    points_event: (N, 2) array of projected 2D points in the event image.
    """
    # Adjust extrinsics to match the convention in visualize_rgb_event
    if rect_matrices is None:
        rect_matrices = dict(rgb=np.eye(3), event=np.eye(3))
    r_rect_rgb = np.eye(4)
    r_rect_rgb[:3, :3] = rect_matrices['rgb']
    r_rect_evt = np.eye(4)
    r_rect_evt[:3, :3] = rect_matrices['event']

    extrinsics = r_rect_rgb @ extrinsics @ np.linalg.inv(
        r_rect_evt)
    R = extrinsics[:3, :3]

    # Compute the inverse projection matrix
    proj_matrix = np.linalg.inv(K_rgb @ R @ np.linalg.inv(K_ev))

    # Convert points to homogeneous coordinates
    points_rgb_hom = np.hstack([points_rgb, np.ones((points_rgb.shape[0], 1))])

    # Map points to event image coordinates
    points_event_hom = (proj_matrix @ points_rgb_hom.T).T

    # Normalize
    points_event = points_event_hom[:, :2] / points_event_hom[:, 2, None]
    return points_event


def compute_pnp_transform(_2d_pts: list, _3d_pts: list, K: np.ndarray, U: np.ndarray = None, rect_mat: np.ndarray = None):
    """
    Computes a transformation matrix between the lidar sensor and camera sensor.

    Parameters:
    -----------
    _2d_pts: set of 2D points on the image plane.
    _3d_pts: set of 3D points on the lidar plane.
    K: intrinsic matrix of camera sensor.
    U: Unification matrix (transforms from camera co-ordinate system to lidar co-ordinate system)
    returns: None | tuple(transformation_matrix, unification_matrix)
    """

    if len(_2d_pts) >= 4 and len(_3d_pts) >= 4:
        points_2d = np.array(_2d_pts, dtype=np.float32)
        points_3d = np.array(_3d_pts, dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(points_3d, points_2d, K, None)

        if success:
            if rect_mat is None:
                rect_mat = np.eye(3)
            if U is None:
                U = np.eye(3)

            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R@rect_mat
            T[:3, 3] = tvec.flatten()
            print("Extrinsic Transformation Matrix:")
            print(T)

            um = np.eye(4)
            um[:3, :3] = U
            T_lidar_to_cam = T@np.linalg.inv(um)
            return T_lidar_to_cam, um
        else:
            print("Error: Unable to compute transformation.")
    else:
        print("Error: Select at least 4 point correspondences.")
    return None


def visualize_projection(image: np.ndarray, points_3d: np.ndarray,
                         points_2d: np.ndarray, intensities: Optional[np.ndarray] = None, depth_dim: int = 0,
                         alpha: float = 0.5,
                         color_map: int = cv2.COLORMAP_JET, debug=False, ) -> np.ndarray:
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

    valid_mask = points_3d[:, depth_dim] > 0
    points_3d = points_3d[valid_mask]
    points_2d = points_2d[valid_mask]
    if debug:
        print('points_3d, points_2d shape:', len(points_3d), len(points_2d))
    overlay = image.copy()
    if intensities is None:
        painting_values = points_3d[:, depth_dim]
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
            cv2.circle(overlay, (x, y), 3, color, -1)
    image = cv2.addWeighted(overlay, alpha, image, 1-alpha, 0)
    return image


def normalize_pixels(points, K, R_rect=None):
    """
    pixel coordinates --> normalized coordinates using Intrinsic matrix.
    """
    pts_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    if R_rect is None:
        R_rect = np.eye(3)
    pts_norm = (R_rect@np.linalg.inv(K)@pts_hom.T).T
    return pts_norm[:, :2]


def visualize_rgb_event(evt_img, rgb_img, K_ev, K_rgb, extrinsics, rect_matrices=None):

    if rect_matrices is None:
        rect_matrices = dict(rgb=np.eye(3), event=np.eye(3))

    # extrinsics = DSEC_T_GT
    r_rect_rgb = np.eye(4)
    r_rect_rgb[:3, :3] = rect_matrices['rgb']
    r_rect_evt = np.eye(4)
    r_rect_evt[:3, :3] = rect_matrices['event']

    extrinsics = r_rect_rgb@extrinsics@np.linalg.inv(r_rect_evt)
    # print('Final Transformation Used:')
    # print(extrinsics)
    R = extrinsics[:3, :3]
    proj_matrix = K_rgb @ R @ np.linalg.inv(K_ev)
    ht, wd, _ = evt_img.shape

    # coords: ht, wd, 2
    coords = np.stack(np.meshgrid(np.arange(wd), np.arange(ht)), axis=-1)
    # coords_hom: ht, wd, 3
    coords_hom = np.concatenate((coords, np.ones((ht, wd, 1))), axis=-1)
    # mapping: ht, wd, 3
    mapping = (proj_matrix @ coords_hom[..., None]).squeeze()
    # mapping: ht, wd, 2
    mapping = (mapping/mapping[..., -1][..., None])[..., :2]
    mapping = mapping.astype('float32')
    proj_img = cv2.remap(rgb_img, mapping, None, interpolation=cv2.INTER_CUBIC)
    evt_img = evt_img.astype(np.uint8)
    red_mask = (evt_img[:, :, 0] > 150) & (evt_img[:, :, 1] < 100) & (evt_img[:, :, 2] < 100)
    blue_mask = (evt_img[:, :, 0] < 100) & (evt_img[:, :, 1] < 100) & (evt_img[:, :, 2] > 150)
    proj_img[red_mask] = evt_img[red_mask]
    proj_img[blue_mask] = evt_img[blue_mask]
    proj_img = cv2.cvtColor(proj_img, cv2.COLOR_BGR2RGB)

    return proj_img
