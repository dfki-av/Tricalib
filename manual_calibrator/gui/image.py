

# third-party imports
import numpy as np
import imageio.v2 as imageio
from PyQt6.QtWidgets import (QVBoxLayout, QDialog, QPushButton, QSlider, QFormLayout,
                             QLabel, QFileDialog, QHBoxLayout, QDoubleSpinBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QIcon

# internal imports
from manual_calibrator.utils.projection import project_points, visualize_projection, visualize_rgb_event


class ImageViewer(QDialog):
    """Secondary window for displaying the image."""

    def __init__(self, image, point_cloud, extrinsics, intrinsics):
        super().__init__()
        self.setWindowTitle("Projection Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon('./data/icons/start_logo.webp'))

        # Layout

        self.image = image
        self.base_image = image.copy()
        self.point_cloud = point_cloud
        self.retrive_info(extrinsics, intrinsics)
        self.project()
        self.initUI()

    def retrive_info(self, extrinsics, intrinsics):
        self.intrinsics = intrinsics
        lidar2cam = np.array(extrinsics['T_lidar_to_cam']['data'])
        cam2img = np.array(extrinsics['T_cam_to_img']['data'])
        self.ext_mat = lidar2cam@cam2img
        self.intrinsics = intrinsics

    def initUI(self):
        "initialize the GUI."

        layout = QVBoxLayout()

        h1_layout = QHBoxLayout()
        h2_layout = QFormLayout()
        


        self.alpha_button = QSlider(Qt.Orientation.Horizontal, self)
        self.alpha_button.setRange(0, 100)
        self.alpha_button.setValue(100)
        self.alpha_button.setFixedWidth(200)
        self.alpha_button.valueChanged.connect(self.on_alpha_changed)

        h2_layout.addRow("Alpha:", self.alpha_button)
        h1_layout.addLayout(h2_layout)

        self.paint_intensity_button = QPushButton("Intensity Mode")
        self.paint_intensity_button.setEnabled(False)
        self.paint_intensity_button.clicked.connect(self.intensity_mode)
        h1_layout.addWidget(self.paint_intensity_button)

        self.paint_depth_button = QPushButton("Depth Mode")
        self.paint_depth_button.clicked.connect(self.depth_mode)
        self.paint_depth_button.setEnabled(True)
        h1_layout.addWidget(self.paint_depth_button)

        self.save_button = QPushButton("Save Projection")
        self.save_button.clicked.connect(self.save_image)
        h1_layout.addWidget(self.save_button)
        h1_layout.setSpacing(10)

        layout.addLayout(h1_layout)
        # QLabel to display the imaged
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_image()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

    def depth_mode(self):
        """Displays the projection with points colorized with depth."""
        self.paint_intensity_button.setEnabled(True)
        self.image = self.base_image.copy()
        self.project(intensity=False)
        self.display_image()
        self.paint_depth_button.setEnabled(False)

    def intensity_mode(self):
        """Displays the projection with points colorized with intensity."""
        self.paint_depth_button.setEnabled(True)
        self.image = self.base_image.copy()
        self.project()
        self.display_image()
        self.paint_intensity_button.setEnabled(False)

    def display_image(self):
        """displays the image in GUI"""
        h, w, ch = self.image.shape
        bytes_per_line = ch * w
        q_image = QImage(self.image.data, w, h,
                         bytes_per_line, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setScaledContents(True)

    def project(self, unification=False, intensity=True):
        """
        projects the point cloud on to image plane and updates the image in GUI.

        Parameters:
        -----------
        unification: deprecated, doesn't reflect in GUI.
        intensity: whether to colorise the projected points with intensity or depth. when intensity False, depth.
        """
        points_3d = self.point_cloud.point.positions.numpy()
        intensities = None
        r_mat = self.ext_mat[:3, :3]
        t_vec = self.ext_mat[:3, 3]
        points_2d = project_points(points_3d=points_3d,
                                   rotation_matrix=r_mat,
                                   translation_vector=t_vec,
                                   camera_matrix=self.intrinsics,
                                   unification=unification)
        if intensity:
            intensities = self.point_cloud.point.intensity.numpy()

        if hasattr(self, 'alpha_button'):
            alpha_value = self.alpha_button.value()/100
        else:
            alpha_value = 1.0
        self.image = visualize_projection(
                self.image, points_3d, points_2d, intensities, alpha_value)

    def save_image(self):
        """save image to disk."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "Image File (*.jpeg; *.png)")
        if file_path:
            imageio.imwrite(file_path, self.image)
    
    def on_alpha_changed(self, value:float):
        self.image = self.base_image.copy()

        if self.paint_intensity_button.isEnabled():
            self.project(intensity=False)
        else:
            self.project()
        self.display_image()

        



class EventImageViewer(QDialog):
    """Secondary window for displaying the image."""

    def __init__(self, evt_image, rgb_image, extrinsics_data):
        super().__init__()
        self.setWindowTitle("Event Projection Viewer")
        self.setGeometry(100, 100, 800, 600)

        # Layout

        self.evt_image = evt_image
        self.rgb_image = rgb_image
        self.extrinsics = np.array(extrinsics_data['T_rgb_evt']['data'])
        self.K_evt = np.array(extrinsics_data["K_evt"]['data'])
        self.K_rgb = np.array(extrinsics_data['K_rgb']['data'])
    
        self.project()
        self.initUI()

    def initUI(self):
        "initialize the GUI."

        layout = QVBoxLayout()

        h1_layout = QHBoxLayout()
    
        self.save_button = QPushButton("Save Projection")
        self.save_button.clicked.connect(self.save_image)
        h1_layout.addWidget(self.save_button)
        h1_layout.setSpacing(10)

        layout.addLayout(h1_layout)
        # QLabel to display the image
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_image()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

    def display_image(self):
        """displays the image in GUI"""
        h, w, ch = self.image.shape
        bytes_per_line = ch * w
        q_image = QImage(self.image.data, w, h,
                         bytes_per_line, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setScaledContents(True)

    def project(self):
        """Projects the event image onto the RGB image using the extrinsics."""
        self.image = visualize_rgb_event(self.evt_image, self.rgb_image,
                            self.K_evt, self.K_rgb, self.extrinsics)
    def save_image(self):
        """save image to disk."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "Image File (*.jpeg; *.png)")
        if file_path:
            imageio.imwrite(file_path, self.image)

class EventLidarViewer(ImageViewer):

    def __init__(self, image, point_cloud, extrinsics, intrinsics, event_rect_matrix):
        self.event_rect_matrix = event_rect_matrix
        super().__init__(image, point_cloud, extrinsics, intrinsics)

    
    def retrive_info(self, extrinsics, intrinsics):
        self.intrinsics = intrinsics
        lidar2cam = np.array(extrinsics['T_lidar_to_evt']['data'])
        cam2evt = np.array(extrinsics['T_cam_to_evt']['data'])
        self.ext_mat = lidar2cam@cam2evt
        self.intrinsics = intrinsics
