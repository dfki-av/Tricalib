__author__ = "Rahul Jakkamsetty"
__license__ = "CC BY-NC-SA 4.0"
__doc__ = """
TriCalib - calib mixin
=======================================
Contains the calibration mixin which supports various calibration related methods.
Developed at DFKI (German Research Center for AI), March 2026.
"""
# python imports


# third-party imports
import numpy as np
import cv2
from PyQt6.QtWidgets import QMessageBox

# internal imports
from tricalib.utils.constants import BASIS_MATRIX, DSEC_R_RECT_EVENT, DSEC_R_RECT_RGB
from tricalib.optim.optimizer import reprojection_error, optimize_calibration
from tricalib.misc import matrices_to_params
from tricalib.utils.io import serialize_dict
from tricalib.utils.projection import normalize_pixels, compute_pnp_transform
from tricalib.gui.secgui import ReprojectionErrorWindow

class CalibrationMixin:
    def __init__(self):
        pass

    def _warn_if_poor_spread(self, points_2d, label=""):
        """Warns if 2D points don't span at least 10% of the image in both axes."""
        if len(points_2d) < 4 or self.image is None:
            return
        pts = np.array(points_2d)
        img_h, img_w = self.image.shape[:2]
        spread_x = (pts[:, 0].max() - pts[:, 0].min()) / img_w
        spread_y = (pts[:, 1].max() - pts[:, 1].min()) / img_h
        if spread_x < 0.1 or spread_y < 0.1:
            QMessageBox.warning(self, "Point Spread Warning",
                f"{label}Points are clustered in a small region.\n"
                "Spread them across the image for better calibration accuracy.")

    def compute_rp_e(self):
        if not self.assert_loaded():
            return

        if self.auto_axis_alignment:
            unification = BASIS_MATRIX
        else:
            unification = np.eye(3)

        if self.rotation_rectification:
            rect_matrices = dict(rgb=DSEC_R_RECT_RGB,
                                 event=DSEC_R_RECT_EVENT)
        else:
            rect_matrices = dict(rgb=np.eye(3),
                                 event=np.eye(3))

        params = matrices_to_params(self._extrinsic_data)
        errors_dict = reprojection_error(params, self.selected_3d_points, self.selected_2d_points, self.selected_ev_points,
                                         self.rgb_camera_matrix, self.evt_camera_matrix,
                                         unification=unification, rect_matrics=rect_matrices, return_errors=True)
        for k in errors_dict:
            errors_dict[k] = np.round(np.abs(errors_dict[k].mean()), 4)
        dlg = ReprojectionErrorWindow(errors_dict, self)
        dlg.show()

    def compute_evt_rgb_transform(self):

        if not self.assert_loaded(flags=['event_image', 'image', 'intrinsics']):
            return

        self._warn_if_poor_spread(self.selected_2d_points, "RGB ")
        self._warn_if_poor_spread(self.selected_ev_points, "Event ")
        if len(self.selected_2d_points) >= 4 and len(self.selected_ev_points) >= 4:
            try:
                points_rgb = np.array(self.selected_2d_points, dtype=np.float32)
                points_evt = np.array(self.selected_ev_points, dtype=np.float32)
                points_rgb_norm = normalize_pixels(
                    points_rgb, self.rgb_camera_matrix)
                points_evt_norm = normalize_pixels(
                    points_evt, self.evt_camera_matrix)
                E, mask = cv2.findEssentialMat(
                    points_rgb_norm, points_evt_norm, method=cv2.RANSAC, prob=0.999, threshold=1e-3, maxIters=10000)

                out = cv2.recoverPose(
                    E=E, points1=points_rgb_norm, points2=points_evt_norm, mask=mask)
                T_rgb_evt = np.eye(4)
                T_rgb_evt[:3, :3] = out[1]
                T_rgb_evt[:3, 3] = out[2].flatten()

                rgb_evt_T_data = dict(T_rgb_to_evt=T_rgb_evt)
                rgb_evt_T_data = serialize_dict(rgb_evt_T_data)

                self._extrinsic_data.update(rgb_evt_T_data)
            except Exception as e:
                QMessageBox.critical(self, "Calibration Error",
                                     f"Failed to compute RGB\u2194Event transformation:\n{e}")
                return
        else:
            QMessageBox.critical(self, "Error", "Select at least 4 point correspondences.")
        self._update_results_panel()

    def compute_pc_evt_transform(self):
        self._warn_if_poor_spread(self.selected_ev_points, "Event ")

        if self.rotation_rectification:
            rect = DSEC_R_RECT_EVENT
        else:
            rect = None

        if self.auto_axis_alignment:
            basis = BASIS_MATRIX
        else:
            basis = None
        if not self.assert_loaded(flags=['pc', 'event_image', 'intrinsics']):
            return
        output = compute_pnp_transform(self.selected_ev_points,
                                       self.selected_3d_points,
                                       self.evt_camera_matrix, basis, rect)

        if output is not None:
            T_lidar_to_evt, _ = output
            evt_lidar_T_data = serialize_dict(
                dict(T_lidar_to_evt=T_lidar_to_evt))
            self._extrinsic_data.update(evt_lidar_T_data)
            self.switch.setEnabled(False)
        self._update_results_panel()

    def compute_pc_rgb_transform(self):
        """Computes the transformation matrix from the selected correspondences."""
        self._warn_if_poor_spread(self.selected_2d_points, "RGB ")

        if self.rotation_rectification:
            rect = DSEC_R_RECT_RGB
        else:
            rect = None

        if self.auto_axis_alignment:
            basis = BASIS_MATRIX
        else:
            basis = None
        if not self.assert_loaded(flags=['pc', 'image','intrinsics']):
            return
        output = compute_pnp_transform(self.selected_2d_points,
                                       self.selected_3d_points,
                                       self.rgb_camera_matrix, basis, rect)

        if output is not None:
            T_lidar_to_cam, _ = output
            rgb_lidar_T_data = serialize_dict(
                {"T_lidar_to_rgb": T_lidar_to_cam})
            self._extrinsic_data.update(rgb_lidar_T_data)
            self.switch.setEnabled(False)
        self._update_results_panel()

    def compute_all(self):
        """Computes the transfromation matrices of all the modalities simultaneoulsy."""
        self._warn_if_poor_spread(self.selected_2d_points, "RGB ")
        self._warn_if_poor_spread(self.selected_ev_points, "Event ")
        if self.rotation_rectification:
            rect_matrices = dict(rgb=DSEC_R_RECT_RGB,
                                 event=DSEC_R_RECT_EVENT)
        else:
            rect_matrices = dict(rgb=np.eye(3),
                                 event=np.eye(3))

        if self.auto_axis_alignment:
            basis = BASIS_MATRIX
        else:
            basis = None
        if not self.assert_loaded():
            return

        extrinsics = optimize_calibration(points_lidar=self.selected_3d_points,
                                          points_rgb=self.selected_2d_points,
                                          points_event=self.selected_ev_points,
                                          K_rgb=self.rgb_camera_matrix, K_ev=self.evt_camera_matrix, unification=basis, rect_matrices=rect_matrices)
        self._extrinsic_data = serialize_dict(extrinsics)
        self._update_results_panel()

    def assert_loaded(self, flags: list = None):
        if flags is None:
            flags = ['image', 'event_image', 'pc', 'intrinsics']
        if self.image is None and 'image' in flags:
            QMessageBox.critical(self, "Error RGB", "RGB Image not loaded")
            return False
        if not hasattr(self, 'event_image') and 'event_image' in flags:
            QMessageBox.critical(self, "Error Event", "Event Image not loaded")
            return False
        if self.point_cloud is None and 'pc' in flags:
            QMessageBox.critical(self, "Error PC", "Point cloud not loaded")
            return False
        if not self._intrinsics_loaded and 'intrinsics' in flags:
            QMessageBox.critical(self, 'Error Intrinsics',
                                 'Intrinsics not loaded.')
            return False
        return True
