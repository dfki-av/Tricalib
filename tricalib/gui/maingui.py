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

Developed at DFKI (German Research Center for AI), December 2024 – March 2026.
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
import pyvista as pv
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QMessageBox, QToolBar,
                              QSizePolicy, QDockWidget, QPlainTextEdit, QWidget, QPushButton,
                             QWidget, QLabel, QHBoxLayout, QStatusBar,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt, QPoint, QTimer, QSize, QProcess
from PyQt6.QtGui import QPainter, QPen, QColor, QIcon, QAction, QFont, QActionGroup

# internal imports
from tricalib.utils.io import ucode_icon, write_json      
from tricalib.gui.style import (Switch, DARK_STYLESHEET, LIGHT_STYLESHEET,
                                         ICON_COLOR_DARK, ICON_COLOR_LIGHT, themed_icon)
from tricalib.misc import image_to_pixmap
from tricalib.gui.workers import run_pyvista_visualizer
from tricalib.gui.mixins import IOMixin, CalibrationMixin, ProjectionMixin

_system_font = None  # set once in main(), used by toggle_theme()


def _os_prefers_dark(app) -> bool:
    """Return True if the OS colour scheme is Dark; falls back to True on older Qt or unknown."""
    try:
        return app.styleHints().colorScheme() != Qt.ColorScheme.Light
    except AttributeError:
        return True  # Qt < 6.5 — default to dark


class PrimaryWindow(QMainWindow, IOMixin, CalibrationMixin, ProjectionMixin):
    """Main GUI for the TriCalib application"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TriCalib")

        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon('./data/icons/start_logo.webp'))

        # Initialize data structures
        self._dark_mode = _os_prefers_dark(QApplication.instance())
        hints = QApplication.instance().styleHints()
        if hasattr(hints, 'colorSchemeChanged'):
            hints.colorSchemeChanged.connect(self._on_os_color_scheme_changed)
        self.auto_axis_alignment = False
        self.rotation_rectification = False
        self.depth_axis = 'x'
        self._depth_active = False
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
        pc_menu.addSeparator()
        depth_axis_menu = pc_menu.addMenu("Depth Axis")
        depth_axis_group = QActionGroup(self)
        depth_axis_group.setExclusive(True)
        for axis, label in [('x', 'X  (e.g. Velodyne)'), ('y', 'Y  (e.g. Blickfeld)'), ('z', 'Z  (vertical)')]:
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(axis == self.depth_axis)
            act.triggered.connect(lambda _, a=axis: self._set_depth_axis(a))
            depth_axis_group.addAction(act)
            depth_axis_menu.addAction(act)

    
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
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
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
            "to see results here.\n\n"
            "Tip: For DSEC, enable Auto Axis\n"
            "Alignment for Joint Optimization"
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

    def mousePressEvent(self, event):
        """Capture mouse click events in the image viewer."""
        if event.button() == Qt.MouseButton.RightButton:
            # Get the relative click position in the QLabel

            label_pos = self.image_label.mapFromGlobal(
                event.globalPosition().toPoint())
            img_x = label_pos.x()
            img_y = label_pos.y()
            if self.image is not None:
                img_h, img_w = self.image.shape[:2]
                label_w = self.image_label.width()
                label_h = self.image_label.height()

                # Scale used by KeepAspectRatio display
                scale = min(label_w / img_w, label_h / img_h)
                # Offset of the image within the label (letterbox/pillarbox)
                offset_x = (label_w - img_w * scale) / 2
                offset_y = (label_h - img_h * scale) / 2

                img_x = int((img_x - offset_x) / scale)
                img_y = int((img_y - offset_y) / scale)

                # Ensure the click is within bounds
                if 0 <= img_x < img_w and 0 <= img_y < img_h:
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
        # Scale radius so the circle is ~4 display pixels regardless of image resolution
        img_h, img_w = self.image.shape[:2]
        display_scale = min(self.image_label.width() / img_w,
                            self.image_label.height() / img_h)
        radius = max(3, int(4 / display_scale))
        pen_width = max(1, int(2 / display_scale))

        painter = QPainter(self.pixmap)
        pen = QPen(QColor("red"))
        pen.setWidth(pen_width)
        painter.setPen(pen)

        brush = QColor(255, 255, 0, 127)
        painter.setBrush(brush)
        painter.drawEllipse(position, radius, radius)
        text_offset = position + QPoint(radius+1, -(radius+1))
        painter.drawText(text_offset, f"P{p_index}")
        painter.end()
        self._update_display_pixmap()

    def _set_depth_axis(self, axis):
        self.depth_axis = axis
        if self._depth_active:
            self.depth()

    def depth(self):
        """GUI button function. Activates depth mode visualization of point cloud."""
        output = self.assert_loaded(['pc'])
        if not output:
            return
        self._depth_active = True
        points = self.point_cloud.point.positions.numpy()
        cloud = pv.PolyData(points)
        col = {'x': 0, 'y': 1, 'z': 2}[self.depth_axis]
        distances = np.abs(points[:, col])
        distances = distances/np.max(distances)
        cloud['Distance'] = distances
        self.display_pointcloud(cloud, scalar='Distance', cmap='rainbow')

    def intensity(self):
        """GUI button function. Activates intensity mode visualization of point cloud."""
        output = self.assert_loaded(['pc'])
        if not output:
            return
        self._depth_active = False
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

    def _on_os_color_scheme_changed(self, scheme):
        is_dark = scheme != Qt.ColorScheme.Light
        if is_dark == self._dark_mode:
            return  # already matches — user may have manually toggled
        self._dark_mode = is_dark
        self.theme_switch.setChecked(is_dark)
        # stateChanged on theme_switch triggers toggle_theme() automatically

    def toggle_theme(self, state):
        self._dark_mode = (state == Qt.CheckState.Checked.value)
        app = QApplication.instance()
        app.setStyleSheet(DARK_STYLESHEET if self._dark_mode else LIGHT_STYLESHEET)
        app.setFont(_system_font)
        icon_color = ICON_COLOR_DARK if self._dark_mode else ICON_COLOR_LIGHT
        self.undo_action.setIcon(themed_icon(
            './data/icons/undo.svg', icon_color))
        self.error_action.setIcon(themed_icon(
            './data/icons/metrics.svg', icon_color))


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
        self.image_label.setScaledContents(False)
        self.image_label.setPixmap(self.pixmap)
        self._update_display_pixmap()
        self._fit_window_to_screen()

    def _update_display_pixmap(self):
        """Scale self.pixmap to fit the label while maintaining aspect ratio."""
        if not hasattr(self, 'pixmap') or self.pixmap is None:
            return
        scaled = self.pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display_pixmap()
        self._fit_window_to_screen()

    def _fit_window_to_screen(self):
        """Reposition window so it's fully within the screen it's currently on."""
        screen = (self.screen() or QApplication.primaryScreen()).availableGeometry()
        x = max(screen.left(), min(self.x(), screen.right() - self.width()))
        y = max(screen.top(), min(self.y(), screen.bottom() - self.height()))
        if self.x() != x or self.y() != y:
            self.move(x, y)

    def draw_points_on_image(self, points: list, rgb_image: np.ndarray | None = None):
        if rgb_image is not None:
            self.pixmap = image_to_pixmap(rgb_image)
        for idx, point in enumerate(points):
            q_point = QPoint(point[0], point[1])
            self.draw_circle(q_point, idx)
        if not points:
            self._update_display_pixmap()

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
                        proc.join(timeout=3)
                        if proc.is_alive():
                            proc.kill()
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



def main():
    mp.set_start_method('spawn')
    app = QApplication(sys.argv)
    global _system_font
    _system_font = app.font()
    app.setStyleSheet(DARK_STYLESHEET if _os_prefers_dark(app) else LIGHT_STYLESHEET)
    app.setFont(_system_font)
    locale.setlocale(locale.LC_NUMERIC, 'C')
    main_window = PrimaryWindow()
    main_window.show()
    app.exec()
    os._exit(0)


if __name__ == "__main__":
    main()
