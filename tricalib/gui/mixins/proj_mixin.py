__author__ = "Rahul Jakkamsetty"
__license__ = "CC BY-NC-SA 4.0"
__doc__ = """
TriCalib - proj mixin
=======================================
Contains the projection mixin which supports various projection related methods.
Developed at DFKI (German Research Center for AI), March 2026.
"""
# python imports
import multiprocessing as mp


# third-party imports
import numpy as np


# internal imports
from tricalib.utils.constants import BASIS_MATRIX
from tricalib.gui.workers import launch_projection_window
from tricalib.gui.image import ImageViewer, EventLidarViewer, EventImageViewer


class ProjectionMixin:
    def __init__(self):
        pass

    def project_all(self):
        """GUI button function. Opens multiple windows dispalying the projected images of all modalities."""
        self.project_extrinsics_pc_evt()
        self.project_extrinsics_pc_rgb()
        self.project_extrinsics_rgb_ev()

    def project_extrinsics_pc_rgb(self):
        """GUI button function.Opens another windows displaying the projected pointcloud on image."""

        rect_matrix, _ = self._get_rect_matrices()

        if self.auto_axis_alignment:
            axis_aligment = BASIS_MATRIX
        else:
            axis_aligment = None

        process = mp.Process(target=launch_projection_window,
                             kwargs=dict(window=ImageViewer, image=self.base_image.copy(),
                                         point_cloud=self.point_cloud, extrinsics=self._extrinsic_data,
                                         intrinsics=self.rgb_camera_matrix, axis_alignment=axis_aligment,
                                         rect_matrix=rect_matrix, path_list=self.state_dict,
                                         dark_mode=self._dark_mode))

        self.pv_processes.append(process)
        process.start()

    def project_extrinsics_pc_evt(self):

        _, rect_matrix = self._get_rect_matrices()

        if self.auto_axis_alignment:
            axis_alignment = BASIS_MATRIX
        else:
            axis_alignment = None

        process = mp.Process(target=launch_projection_window,
                             kwargs=dict(window=EventLidarViewer, image=self.event_image,
                                         point_cloud=self.point_cloud, extrinsics=self._extrinsic_data,
                                         intrinsics=self.evt_camera_matrix, axis_alignment=axis_alignment,
                                         rect_matrix=rect_matrix, path_list=self.state_dict,
                                         dark_mode=self._dark_mode))

        self.pv_processes.append(process)
        process.start()

    def project_extrinsics_rgb_ev(self):

        r_rgb, r_evt = self._get_rect_matrices()
        rect_matrices = dict(rgb=r_rgb, event=r_evt) if self.rotation_rectification else None

        process = mp.Process(target=launch_projection_window,
                             kwargs=dict(window=EventImageViewer, evt_image=self.event_image,
                                         rgb_image=self.image, extrinsics_data=self._extrinsic_data,
                                         K_evt=self.evt_camera_matrix, K_rgb=self.rgb_camera_matrix,
                                         rect_matrices=rect_matrices, path_list=self.state_dict,
                                         dark_mode=self._dark_mode))

        self.pv_processes.append(process)
        process.start()
