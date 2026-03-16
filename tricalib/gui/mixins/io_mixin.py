__author__ = "Rahul Jakkamsetty"
__license__ = "CC BY-NC-SA 4.0"
__doc__ = """
TriCalib - io mixin
=======================================
Contains the io mixin which supports loading and saving different data types.
Developed at DFKI (German Research Center for AI), March 2026.
"""
# python imports
import os
import json
import multiprocessing as mp

# third-party imports
import cv2
import open3d as o3d
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QApplication
from PyQt6.QtCore import Qt, QThread, pyqtSignal


# internal imports
from tricalib.utils.io import load_json, fxfycxcy_to_matrix, write_json
from tricalib.gui.workers import run_event_data_visualizer


class PointCloudLoader(QThread):
    """Background thread for loading point cloud files without blocking the GUI."""
    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            pcd = o3d.t.io.read_point_cloud(self.file_path, format='auto')
            self.loaded.emit(pcd)
        except Exception as e:
            self.failed.emit(str(e))


class IOMixin:
    def __init__(self):
        pass

    def load_intrinsics(self, file_path=None):
        if file_path == 'pass':
            return
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Intrinsics", "", "JSON File (*.json)")
            if not file_path:
                return
        try:
            data: dict = load_json(file_path)
        except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load intrinsics:\n{e}")
            return
        self.state_dict['intrinsics'] = os.path.relpath(file_path)
        self.evt_camera_matrix = fxfycxcy_to_matrix(
            data.get('event_camera_intrinsic'))
        self.rgb_camera_matrix = fxfycxcy_to_matrix(
            data.get('rgb_camera_intrinsic'))
        self._intrinsics_loaded = True

    def load_pnp_points(self, file_path=None):
        """GUI button function. Loads the correspondence points saved on the disk"""
        if file_path == 'pass':
            return
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Pairwise Points", "", "JSON File (*.json)")
            if not file_path:
                return
        try:
            data = load_json(file_path)
        except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load points:\n{e}")
            return
        self.state_dict['pnp_points'] = os.path.relpath(file_path)
        if 'image_points' in data:
            self.selected_2d_points.extend(data['image_points'])
            if hasattr(self, 'pixmap'):
                self.draw_points_on_image(self.selected_2d_points)
        if 'lidar_points' in data:
            self.selected_3d_points.extend(data['lidar_points'])
            self.parent_conn_lidar.send(("UPDATE", self.selected_3d_points))
        if 'event_points' in data:
            self.selected_ev_points.extend(data['event_points'])
            self.parent_conn_event.send(("LOAD", self.selected_ev_points))
        self._update_points_panel()


    def load_extrinsics(self, file_path=None):
        """GUI button function. Loads the extrinsics file stored on the disk."""

        if file_path == 'pass':
            return

        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Extrinsics", "", "JSON File (*.json)")
            if not file_path:
                return
        try:
            self._extrinsic_data = load_json(file_path)
        except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load extrinsics:\n{e}")
            return
        self.state_dict['extrinsics'] = os.path.relpath(file_path)

        if 'K_evt' not in self._extrinsic_data:
            try:
                self._extrinsic_data['K_evt'] = self.evt_camera_matrix.tolist(
                )
            except Exception as e:
                QMessageBox.warning(self, "Intrinsics Warning",
                                    f"Event camera intrinsics not loaded:\n{e}")
        if 'K_rgb' not in self._extrinsic_data:
            try:
                self._extrinsic_data['K_rgb'] = self.rgb_camera_matrix.tolist(
                )
            except Exception as e:
                QMessageBox.warning(self, "Intrinsics Warning",
                                    f"RGB camera intrinsics not loaded:\n{e}")
        self.switch.setEnabled(False)

    def load_image(self, file_path=None):
        """GUI button function. Loads the image from the disk"""
        # Load an image
        if file_path == 'pass':
            return

        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Image", "", "Images (*.png *.jpg)")
            if not file_path:
                return
        self.state_dict['rgb_image'] = os.path.relpath(file_path)
        self.image = cv2.imread(file_path)
        self.image_label.setStatusTip(os.path.basename(file_path))
        self.display_image()

    def load_event_image(self, file_path=None):

        if file_path == 'pass':
            return

        if not file_path:

            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Event Image", "", "Event Images (*.png *.jpg)")
            if not file_path:
                return
        self.state_dict['event_image'] = os.path.relpath(file_path)
        self.event_image = cv2.imread(file_path)
        process = mp.Process(target=run_event_data_visualizer, args=(
            self.child_conn_event, self.event_image, file_path, self._dark_mode))
        self.pv_processes.append(process)
        process.start()
        self.start_ev_timer()

    def load_pointcloud(self, file_path=None):
        """GUI button function. Loads the point cloud from the disk."""
        if file_path == 'pass':
            return
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Point Cloud", "", "Point Clouds (*.pcd)")
            if not file_path:
                return
        self.state_dict['point_cloud'] = os.path.relpath(file_path)
        self._pc_loader = PointCloudLoader(file_path)
        self._pc_loader.loaded.connect(self._on_pointcloud_loaded)
        self._pc_loader.failed.connect(
            lambda msg: QMessageBox.critical(self, "Load Error", f"Failed to load point cloud:\n{msg}"))
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._pc_loader.start()

    def _on_pointcloud_loaded(self, pcd):
        """Slot called when PointCloudLoader finishes."""
        QApplication.restoreOverrideCursor()
        self.point_cloud = pcd
        self.intensity()

    def load_state_button(self):
        self.load_state('read')

    def load_state(self, path=None):
        if path == 'read':
            path, _ = QFileDialog.getOpenFileName(
                self, "Load State file", "", "JSON File (*.json)")

        if path is None:
            fpaths = [i for i in os.listdir('.') if i.startswith('state_dict')]
            if fpaths:
                path = fpaths[0]
            else:
                return

        try:
            self.state_dict = load_json(path)
        except FileNotFoundError:
            return
        # Normalize paths for cross-platform compatibility (Windows saves backslashes)
        _path_keys = ['rgb_image', 'point_cloud', 'event_image', 'intrinsics', 'pnp_points', 'extrinsics']
        for key in _path_keys:
            if key in self.state_dict and self.state_dict[key] != 'pass':
                self.state_dict[key] = self.state_dict[key].replace('\\', '/')
        if self.state_dict.get('load_state'):
            self.load_image(self.state_dict['rgb_image'])
            self.load_pointcloud(self.state_dict['point_cloud'])
            self.load_event_image(self.state_dict['event_image'])
            self.load_intrinsics(self.state_dict['intrinsics'])
            self.load_pnp_points(self.state_dict['pnp_points'])
            self.load_extrinsics(self.state_dict['extrinsics'])
            if 'auto_axis' in self.state_dict:
                self.auto_axis_alignment = self.state_dict['auto_axis']
                self.switch.setChecked(self.auto_axis_alignment)

    def save_points(self):
        """GUI button function. Saves the selected points to disk in JSON format."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Pairwise Points", "", "JSON File (*.json)")
        if file_path:
            data = dict(image_points=self.selected_2d_points,
                        lidar_points=[i for i in self.selected_3d_points],
                        event_points=self.selected_ev_points)
            try:
                write_json(file_path, data)
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save points:\n{e}")

    def save_extrinsics(self):
        """GUI button function. Saves the calculated extrinsics to the disk."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Extrinsics", "", "JSON File (*.json)")
        if file_path:
            try:
                write_json(file_path, self._extrinsic_data)
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save extrinsics:\n{e}")

    def save_state(self):
        """GUI button function. Saves state to the disk."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save State", "", "JSON File (*.json)")
        if file_path:
            self.state_dict['load_state'] = True
            try:
                write_json(file_path, self.state_dict)
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save state:\n{e}")
