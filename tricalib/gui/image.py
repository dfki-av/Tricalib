__author__ = "Rahul Jakkamsetty"
__license__ = "CC BY-NC-SA 4.0"
__doc__ = """
TFKP-Cal - Projection Viewer Windows
===============================================
Provides secondary windows for visualizing calibration results:

  - :class:`ImageViewer` — overlays a projected LiDAR point cloud on an RGB image.
    Supports intensity/depth colouring, adjustable alpha blending, per-axis depth
    selection, and batch video generation from frame sequences.

  - :class:`EventImageViewer` — projects the RGB image onto the event camera plane
    using the computed extrinsic transformation. Also supports video generation.

  - :class:`EventLidarViewer` — subclass of :class:`ImageViewer` that uses the
    LiDAR→Event transformation instead of LiDAR→RGB.

All viewers run as standalone Qt applications in separate subprocesses, spawned
from the main window via :func:`~manual_calibrator.gui.maingui.launch_projection_window`.

Developed at DFKI (German Research Center for AI), December 2024 – August 2025.
"""

# python imports
import os

# third-party imports
import numpy as np
import imageio.v2 as imageio
from PyQt6.QtWidgets import (QVBoxLayout, QMessageBox, QPushButton, QSlider, QFormLayout, QMainWindow, QWidget, QInputDialog,
                             QLabel, QFileDialog, QHBoxLayout, QComboBox, QListView, QProgressDialog, QApplication)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QIcon


# internal imports
from tricalib.utils.projection import project_points, visualize_projection, visualize_rgb_event
from tricalib.utils.io import read_image, read_point_cloud, ucode_icon
from tricalib.misc import decompose_T


class ImageViewer(QMainWindow):
    """Secondary window for displaying the image."""

    def __init__(self, image, point_cloud, extrinsics, intrinsics, axis_alignment, rect_matrix, path_list):
        super().__init__()
        self.setWindowTitle("Projection Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon('./data/icons/start_logo.webp'))

        # Data setup
        self.image = image
        self.base_image = image.copy()
        self.point_cloud = point_cloud
        self.axis_alignment = axis_alignment
        self.rect_matrix = rect_matrix[:3, :3]
        self.path_list = path_list
        self.retrive_info(extrinsics, intrinsics)
        self.initUI()
        self.project()
        self.display_image()

    def retrive_info(self, extrinsics, intrinsics):
        self.intrinsics = intrinsics
        lidar2cam = np.array(extrinsics['T_lidar_to_rgb'])
        self.ext_mat = lidar2cam
        self.intrinsics = intrinsics

    def initUI(self):
        """Initialize the GUI."""

        # ---- Central widget required for QMainWindow ----
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Top controls
        h1_layout = QHBoxLayout()
        h2_layout = QFormLayout()
        h3_layout = QFormLayout()

        self.alpha_button = QSlider(Qt.Orientation.Horizontal, self)
        self.alpha_button.setRange(0, 100)
        self.alpha_button.setValue(100)
        self.alpha_button.setFixedWidth(200)
        self.alpha_button.valueChanged.connect(self.on_attrib_changed)

        h2_layout.addRow("Alpha:", self.alpha_button)
        h1_layout.addLayout(h2_layout)

        self.depth_option = QComboBox(self)
        view = QListView()
        self.depth_option.setView(view)
        self.depth_option.addItems(['x', 'y', 'z'])
        self.depth_option.setCurrentIndex(0)
        self.depth_option.currentIndexChanged.connect(self.on_attrib_changed)
        self.depth_option.setMaxVisibleItems(2)
        view.setMaximumHeight(80)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        h3_layout.addRow("Depth Dimension:", self.depth_option)
        h1_layout.addLayout(h3_layout)

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

        self.video_button = QPushButton(ucode_icon("\U000025B6\U0000FE0F"), "Generate Video")
        self.video_button.clicked.connect(self.generate_video)
        h1_layout.addWidget(self.video_button)
        h1_layout.setSpacing(10)

        layout.addLayout(h1_layout)

        # QLabel to display the image
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_image()
        layout.addWidget(self.image_label)

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
        """Displays the image in GUI."""
        h, w, ch = self.image.shape
        bytes_per_line = ch * w
        q_image = QImage(self.image.data, w, h,
                         bytes_per_line, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setScaledContents(True)

    def project(self, intensity=True):
        """Projects the point cloud onto image plane and updates the image."""
        points_3d = self.point_cloud.point.positions.numpy()
        intensities = None
        r_mat = self.ext_mat[:3, :3]
        t_vec = self.ext_mat[:3, 3]

        points_2d = project_points(points_3d=points_3d,
                                   rotation_matrix=r_mat,
                                   translation_vector=t_vec,
                                   camera_matrix=self.intrinsics,
                                   unification=self.axis_alignment,
                                   rectification_matrix=self.rect_matrix.T)
        if intensity:
            intensities = self.point_cloud.point.intensity.numpy()

        alpha_value = self.alpha_button.value() / 100
        self.image = visualize_projection(
            self.image, points_3d, points_2d, intensities, self.depth_option.currentIndex(), alpha_value)

    def save_image(self):
        """Save image to disk."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "Image File (*.jpeg; *.png)")
        if file_path:
            imageio.imwrite(file_path, self.image)

    def on_attrib_changed(self):
        """Triggered when user changes alpha or depth axis."""
        self.image = self.base_image.copy()
        use_intensity = not self.paint_intensity_button.isEnabled()
        self.project(intensity=use_intensity)
        self.display_image()

    def generate_video(self):
        """Generate a video from image and point cloud sequence with progress display."""
        R, t = decompose_T(np.array(self.ext_mat))
        fps, ok = QInputDialog.getInt(self, "Set Video FPS", "Enter FPS value:",
                                      value=4, min=1, max=60, step=1)

        if not ok:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "", "Video File (*.mp4; *.mkv)"
        )
        if not file_path:
            return

        # Get image list
        if hasattr(self, 'event_flag'):
            img_dir = os.path.dirname(self.path_list['event_image'])
        else:
            img_dir = os.path.dirname(self.path_list['rgb_image'])
        pc_dir = os.path.dirname(self.path_list['point_cloud'])
        imgext = os.listdir(img_dir)[0].split('.')[-1]
        files = sorted(os.listdir(pc_dir))
        total = len(files)

        if total == 0:
            QMessageBox.warning(self, "Error", "No files found in directory.")
            return

        progress = QProgressDialog("Generating video...", "Cancel", 0, total, self)
        progress.setWindowTitle("Please wait")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumWidth(300)

        av = self.alpha_button.value() / 100

        with imageio.get_writer(file_path, fps=fps) as writer:
            for i, path in enumerate(files, start=1):
                if progress.wasCanceled():
                    QMessageBox.information(self, "Cancelled", "Video generation cancelled.")
                    break

                img_path = os.path.join(img_dir, path.replace('pcd', imgext))
                pc_path = os.path.join(pc_dir, path)

                try:
                    img = read_image(img_path)
                    pc = read_point_cloud(pc_path)
                    points_2d = project_points(pc[:, :3], R, t, self.intrinsics,
                                               self.axis_alignment, self.rect_matrix)
                    use_intensity = not self.paint_intensity_button.isEnabled()
                    if use_intensity:
                        ints = pc[:, 3]
                    else:
                        ints = None
                    proj_img = visualize_projection(
                        img, pc[:, :3], points_2d, ints,
                        self.depth_option.currentIndex(), av
                    )
                    writer.append_data(proj_img)
                except Exception as e:
                    print(f"Error processing {path}: {e}")

                progress.setValue(i)
                QApplication.processEvents()  

        progress.close()

        QMessageBox.information(self, "Done", "✅ Video generation complete!")


class EventImageViewer(QMainWindow):
    """Secondary window for displaying the image."""

    def __init__(self, evt_image, rgb_image, extrinsics_data, K_evt, K_rgb, rect_matrices, path_list):
        super().__init__()
        self.setWindowTitle("Event Projection Viewer")
        self.setGeometry(100, 100, 800, 600)

        # Data setup
        self.evt_image = evt_image
        self.rgb_image = rgb_image
        self.extrinsics = np.array(extrinsics_data['T_rgb_to_evt'])
        self.K_evt = K_evt
        self.K_rgb = K_rgb
        self.rect_matrices = rect_matrices
        self.path_list = path_list

        self.project()
        self.initUI()

    def initUI(self):
        """Initialize the GUI."""

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Top controls
        h1_layout = QHBoxLayout()

        self.save_button = QPushButton("Save Projection")
        self.save_button.clicked.connect(self.save_image)
        h1_layout.addWidget(self.save_button)
    
        
        self.video_button = QPushButton(ucode_icon("\U000025B6\U0000FE0F"), "Generate Video")
        self.video_button.clicked.connect(self.generate_video)
        h1_layout.addWidget(self.video_button)
        h1_layout.setSpacing(10)
    
        layout.addLayout(h1_layout)

        # QLabel to display the image
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_image()
        layout.addWidget(self.image_label)

    def display_image(self):
        """Displays the image in GUI."""
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
                                         self.K_evt, self.K_rgb, self.extrinsics, self.rect_matrices)

    def save_image(self):
        """Save image to disk."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "Image File (*.jpeg; *.png)")
        if file_path:
            imageio.imwrite(file_path, self.image)

    def generate_video(self):
        """Generate a video from image and point cloud sequence with progress display."""
        fps, ok = QInputDialog.getInt(self, "Set Video FPS", "Enter FPS value:",
                                      value=4, min=1, max=60, step=1)

        if not ok:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "", "Video File (*.mp4; *.mkv)"
        )
        if not file_path:
            return

        # Get image list

        img_dir = os.path.dirname(self.path_list['rgb_image'])
        evt_dir = os.path.dirname(self.path_list['event_image'])
        files = sorted(os.listdir(evt_dir))
        total = len(files)
        
        if total == 0:
            QMessageBox.warning(self, "Error", "No files found in directory.")
            return

        progress = QProgressDialog("Generating video...", "Cancel", 0, total, self)
        progress.setWindowTitle("Please wait")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumWidth(300)

        with imageio.get_writer(file_path, fps=fps) as writer:
            for i, path in enumerate(files, start=1):
                if progress.wasCanceled():
                    QMessageBox.information(self, "Cancelled", "Video generation cancelled.")
                    break

                img_path = os.path.join(img_dir, path)
                evt_path = os.path.join(evt_dir, path)

                try:
                    rgb_image = read_image(img_path)
                    evt_image = read_image(evt_path)
                    proj_img = visualize_rgb_event(evt_image, rgb_image,
                                         self.K_evt, self.K_rgb, self.extrinsics, self.rect_matrices)
              
                    writer.append_data(proj_img)
                except Exception as e:
                    print(f"Error processing {path}: {e}")

                progress.setValue(i)
                QApplication.processEvents()  

        progress.close()

        QMessageBox.information(self, "Done", "✅ Video generation complete!")


class EventLidarViewer(ImageViewer):

    def __init__(self, image, point_cloud, extrinsics, intrinsics, axis_alignment, rect_matrix, path_list):
        self.event_flag = True

        super().__init__(image, point_cloud, extrinsics,
                         intrinsics, axis_alignment, rect_matrix, path_list)

    def retrive_info(self, extrinsics, intrinsics):
        self.intrinsics = intrinsics
        lidar2cam = np.array(extrinsics['T_lidar_to_evt'])
        self.ext_mat = lidar2cam
        self.intrinsics = intrinsics

