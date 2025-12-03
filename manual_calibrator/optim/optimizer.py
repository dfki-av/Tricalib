__author__ = "Rahul Jakkamsetty"
__license__ = "MIT"
__doc__ = """
Minimization performed across the projection errors of three modalities. 
Developed at DFKI in JULY-AUGUST 2025.
"""


# python imports


# third-party imports
import numpy as np
from scipy.optimize import least_squares


# internal imports
from manual_calibrator.utils.constants import DSEC_R_RECT_EVENT, DSEC_R_RECT_RGB
from manual_calibrator.utils.projection import project_points, project_rgb_to_event
from manual_calibrator.misc import compose_T, quat_to_matrix


def parameters_to_matrices(params: np.ndarray) -> dict:
    """
    Converts the parameters (quaternion form) to matrices and provides a dict of transfromations.
    params: (14/21,) shaped array containing the parameters from the optimizer output

    returns: dict containing transformation matrices.
    """
    R_lidar2rgb = quat_to_matrix(params[:4])
    t_vec_lidar2rgb = params[4:7]
    R_lidar2evt = quat_to_matrix(params[7:11])
    t_vec_lidar2evt = params[11:14]

    # Transformation from lidar to RGB
    T_lidar2rgb = compose_T(R_lidar2rgb, t_vec_lidar2rgb)

    # Transformation from lidar to event
    T_lidar2evt = compose_T(R_lidar2evt, t_vec_lidar2evt)

    # Transformation from rgb to event
    if len(params) > 14:
        R_rgb2evt = quat_to_matrix(params[14:18])
        t_vec_rgb2evt = params[18:21]
        T_rgb2evt = compose_T(R_rgb2evt, t_vec_rgb2evt)
    else:
        T_rgb2evt = np.linalg.inv(T_lidar2rgb)@T_lidar2evt

    out = dict(T_lidar_to_rgb=T_lidar2rgb,
               T_lidar_to_evt=T_lidar2evt,
               T_rgb_to_evt=T_rgb2evt)

    return out


def reprojection_error(params: np.ndarray, points_lidar: list, points_rgb: list, points_event: list,
                       K_rgb: list, K_ev: list, lidar2rgb: dict = None,
                       lidar2evt: dict = None, rgb2event: dict = None,
                       unification: np.ndarray | None = None, rect_matrics: dict = None,
                       return_errors = False) -> np.ndarray | dict:
    """
    Computes the reprojection error for the given parameters.

    Parameters:
    -----------
    params: array of parameters to optimize.
    points_lidar: (N, 3) array of 3D points in LiDAR coordinates.
    points_rgb: (N, 2) array of 2D points in the RGB image.
    points_event: (N, 2) array of 2D points in the event image.
    K_rgb: (3, 3) intrinsic matrix of the RGB camera.
    K_ev: (3, 3) intrinsic matrix of the event camera.
    lidar2rgb: Optional. Pairwise correspondences between lidar and rgb
    lidar2evt: Optional. Pairwise correspondences between lidar and event
    rgb2event: Optional. Pairwise correspondences between rgb and event


    Returns:
    --------
    error: an array of residuals across three modalities.
    """
    rquat_rgb = params[:4]
    tvec_rgb = params[4:7]
    rquat_ev = params[7:11]
    tvec_ev = params[11:14]
    rquat_rgb_ev = params[14:18]
    tvec_rgb_ev = params[18:21]

    if isinstance(K_rgb, list):
        K_rgb = np.array(K_rgb)

    if isinstance(K_ev, list):
        K_ev = np.array(K_ev)

    # Convert rotation quat to rotation matrices
    R_lidar_rgb = quat_to_matrix(rquat_rgb)
    R_lidar_ev = quat_to_matrix(rquat_ev)
    R_rgb_ev = quat_to_matrix(rquat_rgb_ev)

    T_rgb_ev = compose_T(R_rgb_ev, tvec_rgb_ev)

    if lidar2rgb is not None:
        points_lidar_rgb = np.array(points_lidar+lidar2rgb['lidar_points'])
        points_rgb_for_lidar = np.array(points_rgb+lidar2rgb['image_points'])
    else:
        points_lidar_rgb = np.array(points_lidar)
        points_rgb_for_lidar = np.array(points_rgb)

    # Project points from LiDAR to RGB
    points_lidar_projected_to_rgb = project_points(
        points_lidar_rgb, R_lidar_rgb, tvec_rgb, K_rgb, unification=unification, rectification_matrix=rect_matrics['rgb'].T)

    if rgb2event is not None:
        points_rgb_event = np.array(points_rgb+rgb2event['image_points'])
        points_event_for_rgb = np.array(points_event+rgb2event['event_points'])
    else:
        points_rgb_event = np.array(points_rgb)
        points_event_for_rgb = np.array(points_event)

    # Project points from RGB to event
    points_rgb_projected_to_event = project_rgb_to_event(
        points_rgb_event, K_rgb, K_ev, T_rgb_ev, rect_matrices=rect_matrics)

    if lidar2evt is not None:
        points_lidar_evt = np.array(points_lidar+lidar2evt['lidar_points'])
        points_event_for_lidar = np.array(
            points_event+lidar2evt['event_points'])
    else:
        points_lidar_evt = np.array(points_lidar)
        points_event_for_lidar = np.array(points_event)

    points_lidar_projected_to_event = project_points(
        points_lidar_evt, R_lidar_ev, tvec_ev, K_ev, unification=unification, rectification_matrix=rect_matrics['event'].T)

    # Compute the reprojection error

    error_lidar_to_rgb = (points_rgb_for_lidar -
                          points_lidar_projected_to_rgb).ravel()
    error_lidar_to_evt = (points_event_for_lidar -
                          points_lidar_projected_to_event).ravel()
    error_rgb_to_event = (points_event_for_rgb -
                          points_rgb_projected_to_event).ravel()
    if return_errors:
        return dict(error_lidar_to_rgb=error_lidar_to_rgb,
                    error_lidar_to_event=error_lidar_to_evt,
                    error_rgb_to_event=error_rgb_to_event)

    return np.concatenate([error_lidar_to_rgb, error_lidar_to_evt, error_rgb_to_event])


def optimize_calibration(points_lidar: list, points_rgb: list, points_event: list,
                         K_rgb: list | np.ndarray, K_ev: list | np.ndarray, params: np.ndarray = None,
                         lidar2rgb: dict = None, lidar2evt: dict = None, rgb2evt: dict = None,
                         unification: np.ndarray | None = None, rect_matrices: dict = None) -> dict:
    """
    Optimize the calibration parameters using least squares.

    Parameters:
    -----------
    points_lidar: (N, 3) array of 3D points in LiDAR coordinates.
    points_rgb: (N, 2) array of 2D points in the RGB image.
    points_event: (N, 2) array of 2D points in the event image.
    K_rgb: (3, 3) intrinsic matrix of the RGB camera.
    K_ev: (3, 3) intrinsic matrix of the event camera.

    lidar2rgb: Optional. Pairwise correspondences between lidar and rgb
    lidar2evt: Optional. Pairwise correspondences between lidar and event
    rgb2event: Optional. Pairwise correspondences between rgb and event

    Returns:
    --------
    result: dict of transformation matrices.
    """
    if params is None:
        params = np.zeros(21)
        params[3] = 1
        params[10] = 1
        params[17] = 1
    result = least_squares(
        reprojection_error, x0=params,
        args=(points_lidar, points_rgb, points_event,
              K_rgb, K_ev, lidar2rgb, lidar2evt, rgb2evt, unification, rect_matrices),
        method='lm', verbose=2)

    extrinsics = parameters_to_matrices(result.x)

    return extrinsics
