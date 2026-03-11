__author__ = "Rahul Jakkamsetty"
__license__ = "CC BY-NC-SA 4.0"
__doc__ = """
TriCalib - Primary GUI Module
=======================================
Provides the main application window (PrimaryWindow) for the TriCalib tool,
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
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QMessageBox, QToolBar,
                              QSizePolicy, QDockWidget, QPlainTextEdit, QWidget, QPushButton,
                             QWidget, QLabel, QFileDialog, QHBoxLayout, QStatusBar,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt, QPoint, QTimer, QSize, QProcess
from PyQt6.QtGui import QPainter, QPen, QColor, QIcon, QAction, QFont

# internal imports
from tricalib.utils.io import (write_json, load_json, ucode_icon,
                                        fxfycxcy_to_matrix, serialize_dict)
from tricalib.utils.projection import normalize_pixels, compute_pnp_transform
from tricalib.utils.constants import DSEC_R_RECT_EVENT, BASIS_MATRIX, DSEC_R_RECT_RGB
from tricalib.gui.image import ImageViewer, EventImageViewer, EventLidarViewer
from tricalib.gui.secgui import SecondaryWindow, ReprojectionErrorWindow
from tricalib.gui.style import (Switch, DARK_STYLESHEET, LIGHT_STYLESHEET,
                                         ICON_COLOR_DARK, ICON_COLOR_LIGHT, themed_icon)
from tricalib.optim.optimizer import optimize_calibration, reprojection_error
from tricalib.misc import image_to_pixmap, matrices_to_params


class PrimaryWindow(QMainWindow):
    """Main GUI for the TriCalib application"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TriCalib")

        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon('./data/icons/start_logo.webp'))

        # Initialize data structures
        self._dark_mode = True
        self.auto_axis_alignment = False
        self.rotation_rectification = False
        self._intrinsics_loaded = False
        self.image = None
        self.point_cloud = None
        self.selected_2d_points = []
        self.selected_3d_points = []
        self.selected_ev_points = []
        self._extrinsic_data = dict()
        self._points_editing = False
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
        load_data_menu = file_menu.addMenu(
            ucode_icon("\U0001F4E4"), "Load &Data")
        load_data_menu.addAction(load_rgb)
        load_data_menu.addAction(load_pc)
        load_data_menu.addAction(load_evt)

        file_menu.addSeparator()

        # — Configuration —
        file_menu.addAction(load_k)

        file_menu.addSeparator()

        # — Session (load) —
        load_session_menu = file_menu.addMenu(
            ucode_icon("\U0001F4E4"), "Load Se&ssion")
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
        joint_menu = calib_menu.addMenu(
            ucode_icon("\U0001F4BB"), "&Joint (All)")
        joint_menu.addAction(compute_all)
        joint_menu.addSeparator()
        joint_menu.addAction(project_all)

        calib_menu.addSeparator()

        # Group 2 — RGB ↔ Event
        rgb_ev_menu = calib_menu.addMenu(ucode_icon(
            "\U0001F4F7\U000026A1"), "RGB \u2194 Event")
        rgb_ev_menu.addAction(compute_rgb_ev)
        rgb_ev_menu.addSeparator()
        rgb_ev_menu.addAction(project_rgb_evt)

        calib_menu.addSeparator()

        # Group 3 — RGB ↔ LiDAR
        rgb_pc_menu = calib_menu.addMenu(ucode_icon(
            "\U0001F4F7\U0001F7E2"), "RGB \u2194 LiDAR")
        rgb_pc_menu.addAction(compute_rgb_pc)
        rgb_pc_menu.addSeparator()
        rgb_pc_menu.addAction(project_pc_rgb)

        calib_menu.addSeparator()

        # Group 4 — Event ↔ LiDAR
        ev_pc_menu = calib_menu.addMenu(ucode_icon(
            "\U000026A1\U0001F7E2 "), "Event \u2194 LiDAR")
        ev_pc_menu.addAction(compute_evt_pc)
        ev_pc_menu.addSeparator()
        ev_pc_menu.addAction(project_pc_evt)

        # ── Point Cloud Menu ───────────────────────────────────────────────────
        pc_menu = menu.addMenu("&Point Cloud")
        intensity = QAction(ucode_icon("\U0001F506"), "&Intensity Mode", self)
        intensity.setStatusTip(
            "Visualize the point cloud coloured by return intensity")
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
        docs_action.setStatusTip(
            "Open the TriCalib documentation in a browser")
        docs_action.triggered.connect(self.open_docs)
        help_menu.addAction(docs_action)

        help_menu.addSeparator()

        restart = QAction(ucode_icon("\U0001F501"), "Restart App", self)
        restart.setStatusTip("Restart the application (clears all data)")
        restart.triggered.connect(self.confirm_restart)
        help_menu.addAction(restart)

        reinit = QAction(ucode_icon("\U0000267B"), "Reinitialize", self)
        reinit.setStatusTip(
            "Save current session state, then restart and restore it")
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
        self.undo_action = QAction(themed_icon(
            './data/icons/undo.svg', ICON_COLOR_DARK), "Undo", self)
        self.undo_action.setStatusTip(
            "Undoes selection of points across RGB, LiDAR and Event camera")
        self.undo_action.triggered.connect(self.undo)
        toolbar.addAction(self.undo_action)
        toolbar.addSeparator()

        self.error_action = QAction(themed_icon(
            './data/icons/metrics.svg', ICON_COLOR_DARK), "Reprojection Error", self)
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
        self._build_results_panel()
        self._build_points_panel()
        # Right dock spans full height; bottom dock sits to the left of it
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        central_widget.setLayout(layout)
        self.setStatusBar(QStatusBar(self))


    def _build_results_panel(self):
        """Creates and docks the persistent extrinsics results panel."""
        self._results_dock = QDockWidget("Calibration Results", self)
        self._results_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self._results_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        # Scrollable text display
        self._results_text = QPlainTextEdit()
        self._results_text.setReadOnly(True)
        font = QFont("Monospace")
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        font.setPointSize(9)
        self._results_text.setFont(font)
        self._results_text.setPlaceholderText(
            "No extrinsics computed yet.\n\n"
            "Run any Calibration → Compute action\n"
            "to see results here."
        )
        vbox.addWidget(self._results_text)

        # Copy to clipboard button
        copy_btn = QPushButton("📋  Copy to Clipboard")
        copy_btn.setToolTip("Copy all extrinsic matrices to the clipboard")
        copy_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(self._results_text.toPlainText())
        )
        vbox.addWidget(copy_btn)

        self._results_dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._results_dock)
        self._results_dock.setMinimumWidth(280)

    def _update_results_panel(self):
        """Refreshes the results panel with the latest extrinsic data."""
        if not self._extrinsic_data:
            return

        lines = []
        # Keys we care about, with friendly labels
        transform_keys = [
            ("T_lidar_to_rgb", "📷  LiDAR → RGB"),
            ("T_lidar_to_evt", "⚡  LiDAR → Event"),
            ("T_rgb_to_evt",   "📷→⚡  RGB → Event"),
        ]

        for key, label in transform_keys:
            if key not in self._extrinsic_data:
                continue
            T = np.array(self._extrinsic_data[key])
            R = T[:3, :3]
            t = T[:3, 3]

            lines.append(f"{'─'*36}")
            lines.append(f"{label}")
            lines.append(f"{'─'*36}")
            lines.append("Rotation (R):")
            for row in R:
                lines.append(f"  [{row[0]:+.6f}  {row[1]:+.6f}  {row[2]:+.6f}]")
            lines.append("Translation (t) [m]:")
            lines.append(f"  [{t[0]:+.6f}  {t[1]:+.6f}  {t[2]:+.6f}]")
            lines.append("")

        if not lines:
            return

        self._results_text.setPlainText("\n".join(lines))

        # Make the dock visible if it was hidden
        self._results_dock.setVisible(True)
        self._results_dock.raise_()

    def _build_points_panel(self):
        """Creates and docks the selected-points table panel."""
        self._points_dock = QDockWidget("Selected Points", self)
        self._points_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self._points_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        # 7 columns: #, 2D-x, 2D-y, 3D-x, 3D-y, 3D-z, EV-x, EV-y
        self._points_table = QTableWidget(0, 8)
        self._points_table.setHorizontalHeaderLabels(
            ["#", "RGB u [px]", "RGB v [px]", "LiDAR X [m]", "LiDAR Y [m]", "LiDAR Z [m]", "EV u [px]", "EV v [px]"]
        )
        self._points_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._points_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._points_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._points_table.verticalHeader().setVisible(False)
        self._points_table.itemChanged.connect(self._on_point_cell_changed)
        vbox.addWidget(self._points_table)

        self._edit_btn = QPushButton("Edit Points")
        self._edit_btn.setCheckable(True)
        self._edit_btn.toggled.connect(self._toggle_points_edit_mode)
        vbox.addWidget(self._edit_btn)

        self._points_dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._points_dock)
        self._points_dock.setMinimumHeight(160)

    def _update_points_panel(self):
        """Refreshes the points table with the current selected points."""
        self._points_table.blockSignals(True)
        n = max(len(self.selected_2d_points),
                len(self.selected_3d_points),
                len(self.selected_ev_points))
        self._points_table.setRowCount(n)

        def cell(value):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            return item

        for i in range(n):
            self._points_table.setItem(i, 0, cell(f'p{i}'))

            if i < len(self.selected_2d_points):
                p = self.selected_2d_points[i]
                self._points_table.setItem(i, 1, cell(f"{p[0]}"))
                self._points_table.setItem(i, 2, cell(f"{p[1]}"))

            if i < len(self.selected_3d_points):
                p = self.selected_3d_points[i]
                self._points_table.setItem(i, 3, cell(f"{p[0]:.4f}"))
                self._points_table.setItem(i, 4, cell(f"{p[1]:.4f}"))
                self._points_table.setItem(i, 5, cell(f"{p[2]:.4f}"))

            if i < len(self.selected_ev_points):
                p = self.selected_ev_points[i]
                self._points_table.setItem(i, 6, cell(f"{p[0]}"))
                self._points_table.setItem(i, 7, cell(f"{p[1]}"))

        self._points_table.blockSignals(False)

    def _toggle_points_edit_mode(self, checked):
        self._points_editing = checked
        trigger = (QTableWidget.EditTrigger.DoubleClicked
                   if checked
                   else QTableWidget.EditTrigger.NoEditTriggers)
        self._points_table.setEditTriggers(trigger)
        self._edit_btn.setText("Done Editing" if checked else "Edit Points")

    def _on_point_cell_changed(self, item):
        if not self._points_editing:
            return
        self._points_table.blockSignals(True)
        row, col = item.row(), item.column()
        try:
            val = float(item.text())
        except ValueError:
            self._update_points_panel()
            self._points_table.blockSignals(False)
            return

        if col in (1, 2) and row < len(self.selected_2d_points):
            p = list(self.selected_2d_points[row])
            p[col - 1] = int(val)
            self.selected_2d_points[row] = tuple(p)
            self.draw_points_on_image(self.selected_2d_points, self.base_image)

        elif col in (3, 4, 5) and row < len(self.selected_3d_points):
            p = list(self.selected_3d_points[row])
            p[col - 3] = val
            self.selected_3d_points[row] = p
            if hasattr(self, 'parent_conn_lidar'):
                self.parent_conn_lidar.send(("UPDATE", self.selected_3d_points))

        elif col in (6, 7) and row < len(self.selected_ev_points):
            p = list(self.selected_ev_points[row])
            p[col - 6] = int(val)
            self.selected_ev_points[row] = tuple(p)
            self.parent_conn_event.send(("UPDATE", self.selected_ev_points))

        self._points_table.blockSignals(False)

    def open_docs(self):
        """GUI button function. Opens the HTML doc of TriCalib in default browser."""
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
                    self.statusBar().showMessage(f"Selected: ({img_x}, {img_y})")
                    self.selected_2d_points.append((img_x, img_y))
                    self.draw_circle(QPoint(img_x, img_y),
                                     len(self.selected_2d_points)-1)
                    self._update_points_panel()

                else:
                    self.statusBar().showMessage("Click is outside the image bounds.")
            else:
                self.statusBar().showMessage("No image loaded.")

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
            self.parent_conn_lidar.send(("UPDATE", self.selected_3d_points))

        if self.selected_ev_points:
            self.selected_ev_points.pop()
            self.parent_conn_event.send(("UNDO",))

        self._update_points_panel()
        self.statusBar().showMessage(
            f"Undo — 2D: {len(self.selected_2d_points)} pts | "
            f"3D: {len(self.selected_3d_points)} pts | "
            f"EV: {len(self.selected_ev_points)} pts"
        )

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
        self.undo_action.setIcon(themed_icon(
            './data/icons/undo.svg', icon_color))
        self.error_action.setIcon(themed_icon(
            './data/icons/metrics.svg', icon_color))

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

        data = load_json(file_path)
        self.state_dict['pnp_points'] = file_path
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

        self._extrinsic_data = load_json(file_path)
        self.state_dict['extrinsics'] = file_path

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
            self.selected_3d_points.append(point)
            self._update_points_panel()

    def ev_poll(self):
        """gets the point from event image viewer at this timer event."""
        while self.parent_conn_event.poll():
            point = self.parent_conn_event.recv()
            self.selected_ev_points.append(point)
            self._update_points_panel()

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

        if not self.assert_loaded(flags=['event_image', 'image']):
            return

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

        if self.rotation_rectification:
            rect = DSEC_R_RECT_EVENT
        else:
            rect = None

        if self.auto_axis_alignment:
            basis = BASIS_MATRIX
        else:
            basis = None
        if not self.assert_loaded(flags=['pc', 'event_image']):
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

        if self.rotation_rectification:
            rect = DSEC_R_RECT_RGB
        else:
            rect = None

        if self.auto_axis_alignment:
            basis = BASIS_MATRIX
        else:
            basis = None
        if not self.assert_loaded(flags=['pc', 'image']):
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
            flags = ['image', 'event_image', 'pc']
        if self.image is None and 'image' in flags:
            QMessageBox.critical(self, "Error RGB", "RGB Image not loaded")
            return False
        if not hasattr(self, 'event_image') and 'event_image' in flags:
            QMessageBox.critical(self, "Error Event", "Event Image not loaded")
            return False
        if self.point_cloud is None and 'pc' in flags:
            QMessageBox.critical(self, "Error PC", "Point cloud not loaded")
            return False
        if not self._intrinsics_loaded:
            QMessageBox.critical(self, 'Error Intrinsics',
                                 'Intrinsics not loaded.')
            return False
        return True

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
    point_actors = []   # list of (sphere_actor, label_actor) for each selected point
    picked_points = []  # local mirror of selected_3d_points

    def _redraw_all():
        for sa, la in point_actors:
            plotter.remove_actor(sa)
            plotter.remove_actor(la)
        point_actors.clear()
        for i, pt in enumerate(picked_points):
            sa = plotter.add_mesh(
                pv.Sphere(radius=0.05, center=pt),
                color="yellow", pickable=False)
            la = plotter.add_point_labels(
                [pt], [f"P{i}"],
                font_size=12, text_color="black",
                shape=None, always_visible=True)
            point_actors.append((sa, la))
        plotter.render()

    def point_picker_callback(picked_point, picker):
        pt = picked_point.tolist()
        picked_points.append(pt)
        conn.send(pt)
        _redraw_all()

    def poll_pipe(step):
        while conn.poll():
            msg = conn.recv()
            if isinstance(msg, tuple) and msg[0] == "UPDATE":
                picked_points.clear()
                picked_points.extend(msg[1])
                _redraw_all()

    plotter = pv.Plotter()
    plotter.add_axes_at_origin()
    plotter.add_mesh(cloud, scalars=scalar, cmap=cmap,
                     point_size=2, render_points_as_spheres=False, pickable=True)
    plotter.enable_point_picking(
        callback=point_picker_callback, show_message=True, use_picker=True)

    _timer_created = [False]

    def _setup_timer(pl):
        if not _timer_created[0]:
            _timer_created[0] = True
            pl.add_timer_event(max_steps=10_000_000, duration=200, callback=poll_pipe)

    plotter.add_on_render_callback(_setup_timer)
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
