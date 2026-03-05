__author__ = "Rahul Jakkamsetty"
__license__ = "CC BY-NC-SA 4.0"
__doc__ = """
TFKP-Cal - Primary GUI Module
=======================================
Provides the main application window (PrimaryWindow) for the Manual Calibrator tool,
a 2D-3D extrinsic calibration utility for multi-modal sensor setups comprising:
  - RGB camera
  - LiDAR point cloud
  - Event camera

Key capabilities:
  - Interactive point correspondence selection via right-click on 2D images and
    3D point cloud viewer.
  - PnP-based pairwise calibration (LiDAR↔RGB, LiDAR↔Event, RGB↔Event).
  - Joint non-linear optimization over all three modality pairs simultaneously.
  - Projection visualization with intensity/depth coloring and video generation.
  - Reprojection error computation and display.
  - Session state persistence (save/load full tool state to JSON).

Developed at DFKI (German Research Center for AI), December 2024 – August 2025.
"""

# python imports
import sys
import os
import locale
import multiprocessing as mp
import webbrowser

# third-party imports
import numpy as np
import cv2
import open3d as o3d
import pyvista as pv
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QMessageBox, QToolBar, QSizePolicy,
                             QWidget, QLabel, QFileDialog, QHBoxLayout, QStatusBar)
from PyQt6.QtCore import Qt, QPoint, QTimer, QSize, QProcess
from PyQt6.QtGui import QPainter, QPen, QColor, QIcon, QAction

# internal imports
from manual_calibrator.utils.io import (write_json, load_json, ucode_icon,
                                        fxfycxcy_to_matrix, serialize_dict)
from manual_calibrator.utils.projection import normalize_pixels, compute_pnp_transform
from manual_calibrator.utils.constants import DSEC_R_RECT_EVENT, BASIS_MATRIX, DSEC_R_RECT_RGB
from manual_calibrator.gui.image import ImageViewer, EventImageViewer, EventLidarViewer
from manual_calibrator.gui.secgui import SecondaryWindow, ReprojectionErrorWindow
from manual_calibrator.gui.style import (Switch, DARK_STYLESHEET, LIGHT_STYLESHEET,
                                         ICON_COLOR_DARK, ICON_COLOR_LIGHT, themed_icon)
from manual_calibrator.optim.optimizer import optimize_calibration, reprojection_error
from manual_calibrator.misc import image_to_pixmap, matrices_to_params


class PrimaryWindow(QMainWindow):
    """Main GUI for the TFKP-Cal application"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TFKP-Cal")

        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon('./data/icons/start_logo.webp'))

        # Initialize data structures
        self._dark_mode = True
        self.auto_axis_alignment = False
        self.rotation_rectification = False
        self.image = None
        self.point_cloud = None
        self.selected_2d_points = []
        self.selected_3d_points = []
        self.selected_ev_points = []
        self._extrinsic_data = dict()
        self.base_image = None
        self.parent_conn_lidar, self.child_conn_lidar = mp.Pipe()
        self.parent_conn_event, self.child_conn_event = mp.Pipe()
        self.pv_processes = []
        self.state_dict = dict(rgb_image='pass',
                               point_cloud='pass',
                               event_image='pass',
                               intrinsics='pass',
                               pnp_points='pass',
                               extrinsics='pass',
                               load_state=False
                               )

        # GUI Layout
        self.initUI()
        self.load_state()
        self.state_dict['load_state'] = False

    def initUI(self):
        """initializes the GUI when instantiated."""
        # Main container widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        ####### HEADER 1 #######

        load_rgb = QAction(ucode_icon("\U0001F4E4"),
                           "Load &RGB \U0001F4F7", self)
        load_rgb.setStatusTip("Load an RGB Image from disk")
        load_rgb.triggered.connect(self.load_image)

        load_pc = QAction(ucode_icon("\U0001F4E4"),
                          "Load Poi&nt Cloud \U0001F7E2", self)
        load_pc.setStatusTip("Load a point cloud from disk")
        load_pc.triggered.connect(self.load_pointcloud)

        load_evt = QAction(ucode_icon("\U0001F4E4"),
                           "Load &Event Image \U000026A1", self)
        load_evt.setStatusTip("Load an event image from disk")
        load_evt.triggered.connect(self.load_event_image)

        load_k = QAction(ucode_icon("\U0001F4E4"),
                         "Load &Intrinsics", self)
        load_k.setStatusTip("Load Event/RGB Camera intrinsic matrix")
        load_k.triggered.connect(self.load_intrinsics)

        load_pts = QAction(ucode_icon("\U0001F4E4"),
                           "Load &Points \U0001F538", self)
        load_pts.setStatusTip("Load already existing points from disk")
        load_pts.triggered.connect(self.load_pnp_points)

        load_calib = QAction(ucode_icon("\U0001F4E4"),
                             f"Load &Calibration \U0001F3AF", self)
        load_calib.setStatusTip("Load existing calibration from disk")
        load_calib.triggered.connect(self.load_extrinsics)

        load_state = QAction(ucode_icon("\U0001F4E4"),
                             f"Load &State \U0001F5C3", self)
        load_state.setStatusTip("Load state of tool from disk")
        load_state.triggered.connect(self.load_state_button)

        save_state = QAction(ucode_icon("\U0001F4BE"),
                             f"Save Stat&e \U0001F5C3", self)
        save_state.setStatusTip("Load state of tool from disk")
        save_state.triggered.connect(self.save_state)

        save_pts = QAction(ucode_icon("\U0001F4BE"),
                           "Save Poin&ts \U0001F538", self)
        save_pts.setStatusTip("Save selected points to disk")
        save_pts.triggered.connect(self.save_points)

        save_calib = QAction(ucode_icon("\U0001F4BE"),
                             "Save Ca&libration \U0001F3AF", self)
        save_calib.setStatusTip("Save Calibration to disk")
        save_calib.triggered.connect(self.save_extrinsics)

        menu = self.menuBar()
        file_menu = menu.addMenu("&File")

        # — Load Data —
        load_data_menu = file_menu.addMenu(ucode_icon("\U0001F4E4"), "Load &Data")
        load_data_menu.addAction(load_rgb)
        load_data_menu.addAction(load_pc)
        load_data_menu.addAction(load_evt)

        file_menu.addSeparator()

        # — Configuration —
        file_menu.addAction(load_k)

        file_menu.addSeparator()

        # — Session (load) —
        load_session_menu = file_menu.addMenu(ucode_icon("\U0001F4E4"), "Load Se&ssion")
        load_session_menu.addAction(load_pts)
        load_session_menu.addAction(load_calib)
        load_session_menu.addAction(load_state)

        file_menu.addSeparator()

        # — Save —
        save_menu = file_menu.addMenu(ucode_icon("\U0001F4BE"), "&Save")
        save_menu.addAction(save_pts)
        save_menu.addAction(save_calib)
        save_menu.addAction(save_state)


        calib_menu = menu.addMenu("&Calibration")

        compute_all = QAction(ucode_icon("\U0001F4BB"), "Compute &All", self)
        compute_all.setStatusTip(
            "Compute the calibration among all modalities simultaneously.")
        compute_all.triggered.connect(self.compute_all)

        project_all = QAction(ucode_icon("\U0001F52E"), "Project &All", self)
        project_all.setStatusTip(
            "Project all modalities using the calculated calibration.")
        project_all.triggered.connect(self.project_all)

        compute_rgb_ev = QAction(ucode_icon("\U0001F4BB"),
                                 "Compute \U0001F4F7 vs. \U000026A1", self)
        compute_rgb_ev.setStatusTip(
            "Compute the calibration between RGB and Event camera")
        compute_rgb_ev.triggered.connect(self.compute_evt_rgb_transform)

        project_rgb_evt = QAction(ucode_icon("\U0001F52E"),
                                  "Project \U0001F4F7 on \U000026A1", self)
        project_rgb_evt.setStatusTip(
            "Project the RGB image onto the event camera plane and visualize")
        project_rgb_evt.triggered.connect(self.project_extrinsics_rgb_ev)

        compute_rgb_pc = QAction(ucode_icon("\U0001F4BB"),
                                 "Compute \U0001F4F7 vs. \U0001F7E2", self)
        compute_rgb_pc.setStatusTip(
            "Compute the calibration between RGB camera and LiDAR")
        compute_rgb_pc.triggered.connect(self.compute_pc_rgb_transform)

        project_pc_rgb = QAction(ucode_icon("\U0001F52E"),
                                 "Project \U0001F7E2 on \U0001F4F7", self)
        project_pc_rgb.setStatusTip(
            "Project the LiDAR point cloud on the RGB image and visualize")
        project_pc_rgb.triggered.connect(self.project_extrinsics_pc_rgb)

        compute_evt_pc = QAction(ucode_icon("\U0001F4BB"),
                                 "Compute \U000026A1 vs. \U0001F7E2", self)
        compute_evt_pc.setStatusTip(
            "Compute the calibration between Event camera and LiDAR")
        compute_evt_pc.triggered.connect(self.compute_pc_evt_transform)

        project_pc_evt = QAction(ucode_icon("\U0001F52E"),
                                 "Project \U0001F7E2 on \U000026A1", self)
        project_pc_evt.setStatusTip(
            "Project the LiDAR point cloud on the Event image and visualize")
        project_pc_evt.triggered.connect(self.project_extrinsics_pc_evt)

        # Group 1 — All modalities
        joint_menu = calib_menu.addMenu(ucode_icon("\U0001F4BB"), "&Joint (All)")
        joint_menu.addAction(compute_all)
        joint_menu.addSeparator()
        joint_menu.addAction(project_all)

        calib_menu.addSeparator()

        # Group 2 — RGB ↔ Event
        rgb_ev_menu = calib_menu.addMenu(ucode_icon("\U0001F4F7\U000026A1"), "RGB \u2194 Event")
        rgb_ev_menu.addAction(compute_rgb_ev)
        rgb_ev_menu.addSeparator()
        rgb_ev_menu.addAction(project_rgb_evt)

        calib_menu.addSeparator()

        # Group 3 — RGB ↔ LiDAR
        rgb_pc_menu = calib_menu.addMenu(ucode_icon("\U0001F4F7\U0001F7E2"), "RGB \u2194 LiDAR")
        rgb_pc_menu.addAction(compute_rgb_pc)
        rgb_pc_menu.addSeparator()
        rgb_pc_menu.addAction(project_pc_rgb)

        calib_menu.addSeparator()

        # Group 4 — Event ↔ LiDAR
        ev_pc_menu = calib_menu.addMenu(ucode_icon("\U000026A1\U0001F7E2 "), "Event \u2194 LiDAR")
        ev_pc_menu.addAction(compute_evt_pc)
        ev_pc_menu.addSeparator()
        ev_pc_menu.addAction(project_pc_evt)

        # ── Point Cloud Menu ───────────────────────────────────────────────────
        pc_menu = menu.addMenu("&Point Cloud")
        intensity = QAction(ucode_icon("\U0001F506"), "&Intensity Mode", self)
        intensity.setStatusTip("Visualize the point cloud coloured by return intensity")
        intensity.triggered.connect(self.intensity)
        depth = QAction(ucode_icon("\U0001F39A"), "&Depth Mode", self)
        depth.triggered.connect(self.depth)
        depth.setStatusTip("Visualize the point cloud coloured by depth")

        pc_menu.addAction(intensity)
        pc_menu.addSeparator()
        pc_menu.addAction(depth)

        # ── Help Menu ──────────────────────────────────────────────────────────
        # Group 1: Documentation
        # Group 2: Restart / Reinitialize
        help_menu = menu.addMenu("&Help")

        docs_action = QAction(ucode_icon("\U0001F4D6"), "&Documentation", self)
        docs_action.setStatusTip("Open the Manual Calibrator documentation in a browser")
        docs_action.triggered.connect(self.open_docs)
        help_menu.addAction(docs_action)

        help_menu.addSeparator()

        restart = QAction(ucode_icon("\U0001F501"), "Restart App", self)
        restart.setStatusTip("Restart the application (clears all data)")
        restart.triggered.connect(self.confirm_restart)
        help_menu.addAction(restart)

        reinit = QAction(ucode_icon("\U0000267B"), "Reinitialize", self)
        reinit.setStatusTip("Save current session state, then restart and restore it")
        reinit.triggered.connect(self.reinitialize)
        help_menu.addAction(reinit)

        theme_corner = QWidget()
        theme_corner_layout = QHBoxLayout(theme_corner)
        theme_corner_layout.setContentsMargins(4, 0, 8, 0)
        theme_corner_layout.setSpacing(4)
        theme_label = QLabel("Dark Mode:")
        self.theme_switch = Switch()
        self.theme_switch.setChecked(self._dark_mode)
        self.theme_switch.stateChanged.connect(self.toggle_theme)
        self.theme_switch.setStatusTip("Toggle between dark and light mode")
        theme_corner_layout.addWidget(theme_label)
        theme_corner_layout.addWidget(self.theme_switch)
        menu.setCornerWidget(theme_corner, Qt.Corner.TopRightCorner)

        toolbar = QToolBar('ToolBar')
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setOrientation(Qt.Orientation.Vertical)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, toolbar)
        self.undo_action = QAction(themed_icon('./data/icons/undo.svg', ICON_COLOR_DARK), "Undo", self)
        self.undo_action.setStatusTip(
            "Undoes selection of points across RGB, LiDAR and Event camera")
        self.undo_action.triggered.connect(self.undo)
        toolbar.addAction(self.undo_action)
        toolbar.addSeparator()

        self.error_action = QAction(themed_icon('./data/icons/metrics.svg', ICON_COLOR_DARK), "Reprojection Error", self)
        self.error_action.setStatusTip(
            "Calculates reprojections error for selected points")
        self.error_action.triggered.connect(self.compute_rp_e)
        toolbar.addAction(self.error_action)
        
        

        horiz_toolbar = QToolBar("Horizon Toolbar")
        horiz_toolbar.setIconSize(QSize(24, 24))
        horiz_toolbar.setOrientation(Qt.Orientation.Horizontal)
        self.addToolBar(horiz_toolbar)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Preferred)
        horiz_toolbar.addWidget(spacer)

        container = QWidget()
        toolbar_layout = QHBoxLayout(container)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(6)

        rotrect_label = QLabel("Stereo Rectification:")
        self.rotrect_switch = Switch()
        self.rotrect_switch.setChecked(self.rotation_rectification)
        self.rotrect_switch.stateChanged.connect(
            self.toggle_rotation_rectification)
        self.rotrect_switch.setStatusTip(
            "When enabled, applies rectification to correct sensor rotation misalignments."
        )

        toolbar_layout.addWidget(rotrect_label)
        toolbar_layout.addWidget(self.rotrect_switch)

        axis_label = QLabel("Auto Axis Alignment:")

        self.switch = Switch()
        self.switch.setChecked(self.auto_axis_alignment)
        self.switch.stateChanged.connect(self.toggle_unification)
        self.switch.setStatusTip(
            "When enabled, the co-ordinate system of lidar and rgb/event camera sensor are auto-aligned.")
        toolbar_layout.addWidget(axis_label)
        toolbar_layout.addWidget(self.switch)
        horiz_toolbar.addWidget(container)

        self.image_label = QLabel("2D Image Viewer")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)
        self.image_label.setMouseTracking(True)
        self.image_label.setStatusTip("2D Image Viewer")

        # layout.setSpacing(5)
        # layout.addStretch()
        central_widget.setLayout(layout)
        self.setStatusBar(QStatusBar(self))

    def open_docs(self):
        """GUI button function. Opens the HTML doc of Manual calibrator in default browser."""
        url = './docs/doc.html'
        abs_url = os.path.abspath(url)
        webbrowser.open(f"file://{abs_url}")

    def project_all(self):
        """GUI button function. Opens multiple windows dispalying the projected images of all modalities."""
        self.project_extrinsics_pc_evt()
        self.project_extrinsics_pc_rgb()
        self.project_extrinsics_rgb_ev()

    def project_extrinsics_pc_rgb(self):
        """GUI button function.Opens another windows displaying the projected pointcloud on image."""

        if self.rotation_rectification:
            rect_matrix = DSEC_R_RECT_RGB
        else:
            rect_matrix = np.eye(3)

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

        if self.rotation_rectification:
            rect_matrix = DSEC_R_RECT_EVENT
        else:
            rect_matrix = np.eye(3)

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

        if self.rotation_rectification:
            rect_matrices = dict(rgb=DSEC_R_RECT_RGB,
                                 event=DSEC_R_RECT_EVENT)
        else:
            rect_matrices = None

        process = mp.Process(target=launch_projection_window,
                             kwargs=dict(window=EventImageViewer, evt_image=self.event_image,
                                         rgb_image=self.image, extrinsics_data=self._extrinsic_data,
                                         K_evt=self.evt_camera_matrix, K_rgb=self.rgb_camera_matrix,
                                         rect_matrices=rect_matrices, path_list=self.state_dict,
                                         dark_mode=self._dark_mode))

        self.pv_processes.append(process)
        process.start()

    def mousePressEvent(self, event):
        """Capture mouse click events in the image viewer."""
        if event.button() == Qt.MouseButton.RightButton:
            # Get the relative click position in the QLabel

            label_pos = self.image_label.mapFromGlobal(
                event.globalPosition().toPoint())
            img_x = label_pos.x()
            img_y = label_pos.y()
            if self.image is not None:
                pixmap_size = self.image_label.pixmap().size()
                label_size = self.image_label.size()

                # Calculate scaling ratio (image might not fill the label fully)
                scale_x = pixmap_size.width() / label_size.width()
                scale_y = pixmap_size.height() / label_size.height()

                img_x = int(img_x * scale_x)
                img_y = int(img_y * scale_y)

                # Calculate the exact pixel coordinates in the image

                pixmap_width = self.pixmap.width()
                pixmap_height = self.pixmap.height()

                # Ensure the click is within bounds
                if 0 <= img_x < pixmap_width and 0 <= img_y < pixmap_height:
                    print(f"Exact image pixel coordinates: ({img_x}, {img_y})")
                    self.selected_2d_points.append((img_x, img_y))
                    self.draw_circle(QPoint(img_x, img_y),
                                     len(self.selected_2d_points)-1)

                else:
                    print("Click is outside the image bounds.")
            else:
                print("No image loaded.")

    def draw_circle(self, position: QPoint, p_index: int):
        """highlights the selected point with numbering in the image viewer.

        Parameters:
        -----------
        position: at which the circle has to be placed.
        p_index: index of the point.
        """

        painter = QPainter(self.pixmap)
        pen = QPen(QColor("red"))
        pen.setWidth(2)
        painter.setPen(pen)
        radius = 3

        brush = QColor(255, 255, 0, 127)
        painter.setBrush(brush)
        painter.drawEllipse(position, radius, radius)
        text_offset = position + QPoint(radius+1, -(radius+1))
        painter.drawText(text_offset, f"P{p_index}")
        painter.end()

        self.image_label.setPixmap(self.pixmap)

    def depth(self):
        """GUI button function. Activates depth mode visualization of point cloud."""
        points = self.point_cloud.point.positions.numpy()
        cloud = pv.PolyData(points)
        distances = np.min(np.abs(points), axis=1)
        distances = distances/np.max(distances)
        cloud['Distance'] = distances
        self.display_pointcloud(cloud, scalar='Distance', cmap='rainbow')

    def intensity(self):
        """GUI button function. Activates intensity mode visualization of point cloud."""

        points = self.point_cloud.point.positions.numpy()
        intensity = self.point_cloud.point.intensity.numpy()
        intensity_norm = intensity/np.max(intensity)
        cloud = pv.PolyData(points)
        cloud['intensity'] = intensity_norm
        self.display_pointcloud(cloud, scalar='intensity', cmap='plasma')

    def undo(self):
        """GUI button function. Undoes simaltaneously the selected 2D and 3D points."""

        if self.selected_2d_points:
            self.selected_2d_points.pop()
            self.draw_points_on_image(self.selected_2d_points, self.base_image)
            self.image_label.setPixmap(self.pixmap)

        if self.selected_3d_points:
            self.selected_3d_points.pop()

        if self.selected_ev_points:
            self.selected_ev_points.pop()
            self.parent_conn_event.send(("UNDO",))

        print('Info: Remaining 2D points: ', self.selected_2d_points)
        print('Info: Remaining 3D points: ', self.selected_3d_points)
        print('Info: Remaining EV points: ', self.selected_ev_points)

    def toggle_rotation_rectification(self, state):
        if state == Qt.CheckState.Checked.value:
            self.rotation_rectification = True
        else:
            self.rotation_rectification = False

    def toggle_unification(self, state):
        if state == Qt.CheckState.Checked.value:
            self.auto_axis_alignment = True
        else:
            self.auto_axis_alignment = False

    def toggle_theme(self, state):
        self._dark_mode = (state == Qt.CheckState.Checked.value)
        if self._dark_mode:
            QApplication.instance().setStyleSheet(DARK_STYLESHEET)
            icon_color = ICON_COLOR_DARK
        else:
            QApplication.instance().setStyleSheet(LIGHT_STYLESHEET)
            icon_color = ICON_COLOR_LIGHT
        self.undo_action.setIcon(themed_icon('./data/icons/undo.svg', icon_color))
        self.error_action.setIcon(themed_icon('./data/icons/metrics.svg', icon_color))

    def load_intrinsics(self, file_path=None):
        if file_path == 'pass':
            return
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Intrinsics", "", "JSON File (*.json)")
            if not file_path:
                return
        data: dict = load_json(file_path)
        self.state_dict['intrinsics'] = file_path
        self.evt_camera_matrix = fxfycxcy_to_matrix(
            data.get('event_camera_intrinsic'))
        self.rgb_camera_matrix = fxfycxcy_to_matrix(
            data.get('rgb_camera_intrinsic'))

    def load_pnp_points(self, file_path=None):
        """GUI button function. Loads the correspondence points saved on the disk"""
        if file_path == 'pass':
            return
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Pairwise Points", "", "JSON File (*.json)")
            if not file_path:
                return

        data = load_json(file_path)
        self.state_dict['pnp_points'] = file_path
        if 'image_points' in data:
            self.selected_2d_points.extend(data['image_points'])
            if hasattr(self, 'pixmap'):
                self.draw_points_on_image(self.selected_2d_points)
        if 'lidar_points' in data:
            self.selected_3d_points.extend(data['lidar_points'])
        if 'event_points' in data:
            self.selected_ev_points.extend(data['event_points'])
            self.parent_conn_event.send(("LOAD", self.selected_ev_points))

    def load_extrinsics(self, file_path=None):
        """GUI button function. Loads the extrinsics file stored on the disk."""

        if file_path == 'pass':
            return

        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Extrinsics", "", "JSON File (*.json)")
            if not file_path :
                return

        self._extrinsic_data = load_json(file_path)
        self.state_dict['extrinsics'] = file_path

        if 'K_evt' not in self._extrinsic_data:
            try:
                self._extrinsic_data['K_evt'] = self.evt_camera_matrix.tolist(
                )
            except Exception as e:
                print(e)
                print('Possibly the event camera intrisics are not loaded.')
                pass
        if 'K_rgb' not in self._extrinsic_data:
            try:
                self._extrinsic_data['K_rgb'] = self.rgb_camera_matrix.tolist(
                )
            except Exception as e:
                print(e)
                print('Possibly the rgb camera intrinsics are not loaded')
                pass
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
        self.state_dict['rgb_image'] = file_path
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
        self.state_dict['event_image'] = file_path
        self.event_image = cv2.imread(file_path)
        process = mp.Process(target=run_event_data_visualizer, args=(
            self.child_conn_event, self.event_image, file_path, self._dark_mode))
        self.pv_processes.append(process)
        process.start()
        self.start_ev_timer()

    def load_pointcloud(self, file_path=None):
        """GUI button function. Loads the point cloud from the disk."""
        # Load a 3D point cloud
        if file_path == 'pass':
            return
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Point Cloud", "", "Point Clouds (*.pcd)")
            if not file_path:
                return
        self.state_dict['point_cloud'] = file_path
        self.point_cloud = o3d.t.io.read_point_cloud(
            file_path, format='auto')
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
        if self.state_dict.get('load_state'):
            self.load_image(self.state_dict['rgb_image'])
            self.load_pointcloud(self.state_dict['point_cloud'])
            self.load_event_image(self.state_dict['event_image'])
            self.load_intrinsics(self.state_dict['intrinsics'])
            self.load_pnp_points(self.state_dict['pnp_points'])
            self.load_extrinsics(self.state_dict['extrinsics'])

    def save_points(self):
        """GUI button function. Saves the selected points to disk in JSON format."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Pairwise Points", "", "JSON File (*.json)")
        if file_path:
            data = dict(image_points=self.selected_2d_points,
                        lidar_points=[i for i in self.selected_3d_points],
                        event_points=self.selected_ev_points)
            write_json(file_path, data)

    def save_extrinsics(self):
        """GUI button function. Saves the calculated extrinsics to the disk."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Extrinsics", "", "JSON File (*.json)")
        if file_path:
            write_json(file_path, self._extrinsic_data)

    def save_state(self):
        """GUI button function. Saves state to the disk."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save State", "", "JSON File (*.json)")
        if file_path:
            self.state_dict['load_state'] = True
            write_json(file_path, self.state_dict)

    def display_pointcloud(self, cloud=None, scalar=None, cmap=None):
        """Instantiates another mp Process to display the loaded point cloud.
        Multiprocessing is required to visualized the point cloud on linux platforms."""
        process = mp.Process(target=run_pyvista_visualizer, args=(
            cloud, scalar, cmap, self.child_conn_lidar))
        self.pv_processes.append(process)
        process.start()
        self.start_pc_timer()

    def display_image(self):
        """Function to display the image once the image is loaded."""
        # Convert the image for PyQt display
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.base_image = rgb_image.copy()
        self.pixmap = image_to_pixmap(rgb_image)
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setScaledContents(True)

    def draw_points_on_image(self, points: list, rgb_image: np.ndarray | None = None):
        if rgb_image is not None:
            self.pixmap = image_to_pixmap(rgb_image)
        for idx, point in enumerate(points):
            q_point = QPoint(point[0], point[1])
            self.draw_circle(q_point, idx)

    def start_ev_timer(self):
        """starts the pyqt timer"""
        self.ev_timer = QTimer(self)
        self.ev_timer.timeout.connect(self.ev_poll)
        self.ev_timer.start(100)

    def start_pc_timer(self):
        self.pc_timer = QTimer(self)
        self.pc_timer.timeout.connect(self.pc_poll)
        self.pc_timer.start(100)

    def pc_poll(self):
        while self.parent_conn_lidar.poll():
            point = self.parent_conn_lidar.recv()
            print("Selected 3D point:", point)
            self.selected_3d_points.append(point)

    def ev_poll(self):
        """gets the point from event image viewer at this timer event."""
        while self.parent_conn_event.poll():
            point = self.parent_conn_event.recv()
            print("Selected EV point:", point)
            self.selected_ev_points.append(point)

    def compute_rp_e(self):

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
        if len(self.selected_2d_points) >= 4 and len(self.selected_ev_points) >= 4:
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
            print('RGB to Event Transformation Matrix:')
            print(T_rgb_evt)
        else:
            print("Error: Select at least 4 point correspondences.")

    def compute_pc_evt_transform(self):

        if self.rotation_rectification:
            rect = DSEC_R_RECT_EVENT
        else:
            rect = None

        if self.auto_axis_alignment:
            basis = BASIS_MATRIX
        else:
            basis = None
        output = compute_pnp_transform(self.selected_ev_points,
                                       self.selected_3d_points,
                                       self.evt_camera_matrix, basis, rect)

        if output is not None:
            T_lidar_to_evt, _ = output
            evt_lidar_T_data = serialize_dict(
                dict(T_lidar_to_evt=T_lidar_to_evt))
            self._extrinsic_data.update(evt_lidar_T_data)
            self.switch.setEnabled(False)

    def compute_pc_rgb_transform(self):
        """Computes the transformation matrix from the selected correspondences."""

        if self.rotation_rectification:
            rect = DSEC_R_RECT_RGB
        else:
            rect = None

        if self.auto_axis_alignment:
            basis = BASIS_MATRIX
        else:
            basis = None

        output = compute_pnp_transform(self.selected_2d_points,
                                       self.selected_3d_points,
                                       self.rgb_camera_matrix, basis, rect)

        if output is not None:
            T_lidar_to_cam, _ = output
            rgb_lidar_T_data = serialize_dict(
                {"T_lidar_to_rgb": T_lidar_to_cam})
            self._extrinsic_data.update(rgb_lidar_T_data)
            self.switch.setEnabled(False)

    def compute_all(self):
        """Computes the transfromation matrices of all the modalities simultaneoulsy."""
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

        extrinsics = optimize_calibration(points_lidar=self.selected_3d_points,
                                          points_rgb=self.selected_2d_points,
                                          points_event=self.selected_ev_points,
                                          K_rgb=self.rgb_camera_matrix, K_ev=self.evt_camera_matrix, unification=basis, rect_matrices=rect_matrices)
        self._extrinsic_data = serialize_dict(extrinsics)

    def closeEvent(self, event):
        """Ensure PyVista process is closed when GUI closes."""

        if not self.state_dict['load_state']:
            if os.path.exists('./state_dict.json'):
                os.remove('./state_dict.json')

        if hasattr(self, "pv_processes"):
            if self.pv_processes:
                for proc in self.pv_processes:
                    if proc.is_alive():
                        proc.terminate()
                        proc.join()
        event.accept()

    def confirm_restart(self, reinit=False):
        reply = QMessageBox.question(self, "Restart", "Restart the app?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if reinit:
                write_json('./state_dict.json', self.state_dict)
            self.restart()

    def restart(self):
        QProcess.startDetached(sys.executable, sys.argv)
        QApplication.quit()

    def reinitialize(self):
        self.state_dict['load_state'] = True
        self.confirm_restart(True)


def run_pyvista_visualizer(cloud, scalar, cmap, conn):
    """separate function to launch on different process so that visualizer runs on a different process. Needed for linux platforms."""
    def point_picker_callback(picked_point, picker):
        conn.send(picked_point.tolist())

    plotter = pv.Plotter()
    plotter.add_axes_at_origin()
    plotter.add_mesh(cloud, scalars=scalar, cmap=cmap,
                     point_size=2, render_points_as_spheres=False, pickable=True)

    plotter.enable_point_picking(
        callback=point_picker_callback, show_message=True, use_picker=True)
    plotter.show()


def run_event_data_visualizer(conn, ev_img, ev_img_path, dark_mode=True):
    app = QApplication([])
    app.setStyleSheet(DARK_STYLESHEET if dark_mode else LIGHT_STYLESHEET)
    sec_wdw = SecondaryWindow(conn, ev_img)
    sec_wdw.image_label.setStatusTip(os.path.basename(ev_img_path))
    sec_wdw.show()
    sec_wdw.display_image()
    sys.exit(app.exec())


def launch_projection_window(**kwargs):
    app = QApplication([])
    dark_mode = kwargs.pop('dark_mode', True)
    app.setStyleSheet(DARK_STYLESHEET if dark_mode else LIGHT_STYLESHEET)
    window = kwargs.pop('window')
    proj_wdw = window(**kwargs)
    proj_wdw.show()
    sys.exit(app.exec())


def main():
    mp.set_start_method('spawn')
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    locale.setlocale(locale.LC_NUMERIC, 'C')
    main_window = PrimaryWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
