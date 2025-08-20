__author__ = "Rahul Jakkamsetty"
__license__ = "MIT"
__doc__ = """
A 2D-3D manual calibration tool from RGB Image and Corresponding Point Cloud. Developed at DFKI DEC-JAN 2024-25.
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
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QDialog, QToolBar, 
                             QPushButton, QWidget, QLabel, QFileDialog, QHBoxLayout, QStatusBar)
from PyQt6.QtCore import Qt, QPoint, QTimer, QSize
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QIcon, QAction

# internal imports
from manual_calibrator.utils.io import write_json, load_json, ucode_icon, fxfycxcy_to_matrix
from manual_calibrator.utils.projection import normalize_pixels, compute_pnp_transform
from manual_calibrator.utils.constants import DSEC_R_RECT_EVENT, BASIS_MATRIX
from manual_calibrator.gui.image import ImageViewer, EventImageViewer, EventLidarViewer
from manual_calibrator.gui.secgui import SecondaryWindow


class PrimaryWindow(QMainWindow):
    """Main GUI for the Manual Calibrator application"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manual Calibrator")

        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon('./data/icons/start_logo.webp'))

        # Initialize data structures
        self.image = None
        self.point_cloud = None
        self.selected_2d_points = []
        self.selected_3d_points = []
        self.selected_ev_points = []
        self._extrinsic_data = dict()
        self.base_image = None
        self.image_backups = []
        self.parent_conn_lidar, self.child_conn_lidar = mp.Pipe()
        self.parent_conn_event, self.child_conn_event = mp.Pipe()
        self.pv_processes = []

        # GUI Layout
        self.initUI()

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
        file_menu.addAction(load_rgb)
        file_menu.addSeparator()
        file_menu.addAction(load_pc)
        file_menu.addSeparator()
        file_menu.addAction(load_evt)
        file_menu.addSeparator()
        file_menu.addAction(load_k)
        file_menu.addSeparator()
        file_menu.addAction(load_pts)
        file_menu.addSeparator()
        file_menu.addAction(load_calib)
        file_menu.addSeparator()
        file_menu.addAction(save_pts)
        file_menu.addSeparator()
        file_menu.addAction(save_calib)

        calib_menu = menu.addMenu("&Calibration")
        compute_rgb_ev = QAction(ucode_icon("\U0001F4BB"),
                                 "&Compute \U0001F4F7 vs. \U000026A1", self)
        compute_rgb_ev.setStatusTip(
            "Compute the Calibration between RGB and Event camera")
        compute_rgb_ev.triggered.connect(self.compute_evt_rgb_transform)
        calib_menu.addAction(compute_rgb_ev)

        calib_menu.addSeparator()
        project_rgb_evt = QAction(ucode_icon("\U0001F52E"),
                          "&Project \U0001F4F7 on \U000026A1", self)
        project_rgb_evt.setStatusTip(
            "project the RGB on event data using calibration  and visualize")
        project_rgb_evt.triggered.connect(self.project_extrinsics_rgb_ev)
        calib_menu.addAction(project_rgb_evt)

        calib_menu.addSeparator()

        compute_rgb_pc = QAction(ucode_icon("\U0001F4BB"),
                                 "&Compute \U0001F4F7 vs. \U0001F7E2", self)
        compute_rgb_pc.setStatusTip(
            "Compute the Calibration between RGB and Point Cloud")
        compute_rgb_pc.triggered.connect(self.compute_pc_rgb_transform)
        calib_menu.addAction(compute_rgb_pc)
        calib_menu.addSeparator()

        project_pc_rgb = QAction(ucode_icon("\U0001F52E"),
                          "&Project \U0001F7E2 on \U0001F4F7", self)
        project_pc_rgb.setStatusTip(
            "project the point cloud on RGB using calibration  and visualize")
        project_pc_rgb.triggered.connect(self.project_extrinsics_pc_rgb)
        calib_menu.addAction(project_pc_rgb)

        calib_menu.addSeparator()

        compute_evt_pc = QAction(ucode_icon("\U0001F4BB"),
                                 "&Compute \U000026A1 vs \U0001F7E2", self)
        compute_evt_pc.setStatusTip(
            "Compute the Calibration between Event camera and Point Cloud")
        compute_evt_pc.triggered.connect(self.compute_pc_evt_transform)
        calib_menu.addAction(compute_evt_pc)
        calib_menu.addSeparator()

        project_pc_evt = QAction(ucode_icon("\U0001F52E"),
                          "&Project \U0001F7E2  on \U000026A1", self)
        project_pc_evt.setStatusTip(
            "project the point cloud on Event image using calibration and visualize")
        project_pc_evt.triggered.connect(self.project_extrinsics_pc_evt)
        calib_menu.addAction(project_pc_evt)


        pc_menu = menu.addMenu("&Point Cloud")
        intensity = QAction(ucode_icon("\U0001F506"), "&Intensity Mode", self)
        intensity.setStatusTip("Visualize the point cloud in intensity mode")
        intensity.triggered.connect(self.intensity)
        depth = QAction(ucode_icon("\U0001F39A"), "&Depth Mode", self)
        depth.triggered.connect(self.depth)
        depth.setStatusTip("Visualize the point cloud in depth mode")

        pc_menu.addAction(intensity)
        pc_menu.addSeparator()
        pc_menu.addAction(depth)

        toolbar = QToolBar('ToolBar')
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setOrientation(Qt.Orientation.Vertical)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, toolbar)
        undo_action = QAction(ucode_icon("\U000021A9"), "undo", self)
        undo_action.setStatusTip(
            "Undoes selection of points across RGB, LiDAR and Event camera")
        undo_action.triggered.connect(self.undo)
        toolbar.addAction(undo_action)

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

    def project_extrinsics_pc_rgb(self):
        """GUI button function.Opens another windows displaying the projected pointcloud on image."""
        imageviewer = ImageViewer(self.base_image.copy(),
                                  self.point_cloud,
                                  self._extrinsic_data,
                                  self.rgb_camera_matrix)
        imageviewer.exec()

    def project_extrinsics_pc_evt(self):

        imageviewer = EventLidarViewer(self.event_image.copy(),
                                  self.point_cloud,
                                  self._extrinsic_data,
                                  self.evt_camera_matrix, DSEC_R_RECT_EVENT)
        imageviewer.exec()

    def project_extrinsics_rgb_ev(self):
        imageviewer = EventImageViewer(self.event_image, self.image,
                                       self._extrinsic_data)
        imageviewer.exec()

    def mousePressEvent(self, event):
        """Capture mouse click events in the image viewer."""
        if event.button() == Qt.MouseButton.RightButton:
            # Get the relative click position in the QLabel

            label_pos = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
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
                    self.image_backups.append(self.pixmap.copy())
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
        radius = 5

        brush = QColor("red")
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
            img = self.image_backups.pop()
            self.image_label.setPixmap(img)
            self.pixmap = img

        if self.selected_3d_points:
            self.selected_3d_points.pop()

        if self.selected_ev_points:
            self.selected_ev_points.pop()
            self.parent_conn_event.send(("UNDO",))

        print('Info: Remaining 2D points: ', self.selected_2d_points)
        print('Info: Remaining 3D points: ', self.selected_3d_points)
        print('Info: Remaining EV points: ', self.selected_ev_points)

  
    def load_intrinsics(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Intrinsics", "", "JSON File (*.json)")
        if file_path:
            data:dict = load_json(file_path)
            self.evt_camera_matrix = fxfycxcy_to_matrix(
                data.get('event_camera_intrinsic'))
            self.rgb_camera_matrix = fxfycxcy_to_matrix(
                data.get('rgb_camera_intrinsic'))

    def load_pnp_points(self):
        """GUI button function. Loads the correspondence points saved on the disk"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Pairwise Points", "", "JSON File (*.json)")
        if file_path:
            data = load_json(file_path)
            if 'image_points' in data:
                self.selected_2d_points.extend(data['image_points'])
                if hasattr(self, 'pixmap'):
                    for i, point in enumerate(self.selected_2d_points):
                        self.draw_circle(QPoint(*point), i)
            if 'lidar_points' in data:
                self.selected_3d_points.extend(data['lidar_points'])
            if 'event_points' in data:
                self.selected_ev_points.extend(data['event_points'])
                self.parent_conn_event.send(("LOAD", self.selected_ev_points))

    def load_extrinsics(self):
        """GUI button function. Loads the extrinsics file stored on the disk."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Extrinsics", "", "JSON File (*.json)")
        if file_path:
            self._extrinsic_data = load_json(file_path)
            self.save_button.setEnabled(True)
            self.project_button.setEnabled(True)

    def load_image(self):
        """GUI button function. Loads the image from the disk"""
        # Load an image
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", "Images (*.png *.jpg)")
        if file_path:
            self.image = cv2.imread(file_path)
            self.image_label.setStatusTip(os.path.basename(file_path))
            self.display_image()

    def load_event_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Event Image", "", "Event Images (*.png *.jpg)")
        if file_path:
            self.event_image = cv2.imread(file_path)
            process = mp.Process(target=run_event_data_visualizer, args=(
                self.child_conn_event, self.event_image, file_path))
            self.pv_processes.append(process)
            process.start()
            self.start_ev_timer()

    def load_pointcloud(self):
        """GUI button function. Loads the point cloud from the disk."""
        # Load a 3D point cloud
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Point Cloud", "", "Point Clouds (*.pcd)")
        if file_path:
            self.point_cloud = o3d.t.io.read_point_cloud(
                file_path, format='auto')
            self.intensity()

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
        """GUI button function. Saves the calculates extrinsics to the disk."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Extrinsics", "", "JSON File (*.json)")
        if file_path:
            write_json(file_path, self._extrinsic_data)

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
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h,
                         bytes_per_line, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setScaledContents(True)

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

    def compute_evt_rgb_transform(self):
        if len(self.selected_2d_points) >= 4 and len(self.selected_ev_points) >= 4:
            points_rgb = np.array(self.selected_2d_points, dtype=np.float32)
            points_evt = np.array(self.selected_ev_points, dtype=np.float32)
            points_rgb_norm = normalize_pixels(
                points_rgb, self.rgb_camera_matrix)
            points_evt_norm = normalize_pixels(
                points_evt, self.evt_camera_matrix)
            E, _ = cv2.findEssentialMat(
                points_rgb_norm, points_evt_norm, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            E = E[:3, :]
            out = cv2.recoverPose(
                E=E, points1=points_rgb_norm, points2=points_evt_norm,)
            T_rgb_evt = np.eye(4)
            T_rgb_evt[:3, :3] = out[1]
            T_rgb_evt[:3, 3] = out[2].flatten()

            rgb_evt_T_data = dict(T_rgb_evt=dict(data=T_rgb_evt.tolist()),
                                  K_evt=dict(
                                      data=self.evt_camera_matrix.tolist()),
                                  K_rgb=dict(data=self.rgb_camera_matrix.tolist()))

            self._extrinsic_data.update(rgb_evt_T_data)
            print('RGB to Event Transformation Matrix:')
            print(T_rgb_evt)
        else:
            print("Error: Select at least 4 point correspondences.")

    def compute_pc_evt_transform(self):
        output = compute_pnp_transform(self.selected_ev_points,
                                       self.selected_3d_points,
                                       self.evt_camera_matrix, BASIS_MATRIX)

        if output is not None:
            T_lidar_to_evt, um = output
            self._extrinsic_data.update({"T_lidar_to_evt":  T_lidar_to_evt.tolist(),
                                        "T_evt_to_img": um.tolist(),
                                         "K_evt": self.evt_camera_matrix.tolist()})

    def compute_pc_rgb_transform(self):
        """Computes the transformation matrix from the selected correspondences."""

        output = compute_pnp_transform(self.selected_2d_points,
                                       self.selected_3d_points,
                                       self.rgb_camera_matrix, BASIS_MATRIX)

        if output is not None:
            T_lidar_to_cam, um = output
            self._extrinsic_data.update({"T_lidar_to_rgb":T_lidar_to_cam.tolist(),
                                         "T_rgb_to_img":  um.tolist(),
                                         "K_rgb": self.rgb_camera_matrix.tolist()})

    def closeEvent(self, event):
        """Ensure PyVista process is closed when GUI closes."""

        if hasattr(self, "pv_processes"):
            if self.pv_processes:
                for proc in self.pv_processes:
                    if proc.is_alive():
                        proc.terminate()
                        proc.join()
        event.accept()


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


def run_event_data_visualizer(conn, ev_img, ev_img_path):
    app = QApplication([])
    sec_wdw = SecondaryWindow(conn, ev_img)
    sec_wdw.image_label.setStatusTip(os.path.basename(ev_img_path))
    sec_wdw.show()
    sec_wdw.display_image()
    sys.exit(app.exec())


def main():
    mp.set_start_method('spawn')
    app = QApplication(sys.argv)
    locale.setlocale(locale.LC_NUMERIC, 'C')
    main_window = PrimaryWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
