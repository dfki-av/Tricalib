

# third-party imports
import numpy as np
import imageio.v2 as imageio
from PyQt5.QtWidgets import (QVBoxLayout, QDialog, QPushButton,
                             QLabel, QFileDialog, QHBoxLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

# internal imports
from manual_calibrator.utils.projection import project_points, visualize_projection


class ImageViewer(QDialog):
    """Secondary window for displaying the image."""

    def __init__(self, image, point_cloud, extrinsics, intrinsics):
        super().__init__()
        self.setWindowTitle("Projection Viewer")
        self.setGeometry(100, 100, 800, 600)

        # Layout

        self.image = image
        self.base_image = image.copy()
        self.point_cloud = point_cloud
        lidar2cam = np.array(extrinsics['T_lidar_to_cam']['data'])
        cam2img = np.array(extrinsics['T_cam_to_img']['data'])
        self.ext_mat = lidar2cam@cam2img
        self.intrinsics = intrinsics
        self.project()
        self.initUI()

    def initUI(self):
        "initialize the GUI."

        layout = QVBoxLayout()

        h1_layout = QHBoxLayout()
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
        # QLabel to display the image
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
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
                         bytes_per_line, QImage.Format_RGB888)
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

        self.image = visualize_projection(
            self.image, points_3d, points_2d, intensities)

    def save_image(self):
        """save image to disk."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "Image File (*.jpeg; *.png)")
        if file_path:
            imageio.imwrite(file_path, self.image)
