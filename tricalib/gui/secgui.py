__author__ = "Rahul Jakkamsetty"
__license__ = "CC BY-NC-SA 4.0"
__doc__ = """
TriCalib - Secondary GUI Module
=========================================
Provides auxiliary windows used by the TriCalib:

  - :class:`SecondaryWindow` — Event camera image viewer that runs in a
    separate process. Accepts right-click point selection and communicates
    selections back to the main window via a multiprocessing Pipe. Also
    receives UNDO and LOAD commands from the main window.

  - :class:`ReprojectionErrorWindow` — Modal dialog that displays the mean
    absolute reprojection error (in pixels) for each sensor-pair combination
    after calibration.

Developed at DFKI (German Research Center for AI), December 2024 – August 2025.
"""

# python imports
from pathlib import Path


# third-party imports
import numpy as np
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QDialog, QToolBar, QFormLayout,
                             QPushButton, QWidget, QLabel, QFileDialog, QHBoxLayout, QStatusBar)
from PyQt6.QtCore import Qt, QPoint, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QIcon

# internal imports
from tricalib.misc import image_to_pixmap



_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class SecondaryWindow(QMainWindow):
    """Secondary Window allocated to visualized Event Data"""

    def __init__(self, child_connection, ev_img):
        super().__init__()
        self.setWindowTitle("Event Image")
        self.setGeometry(200, 200, ev_img.shape[1], ev_img.shape[0])
        self.setWindowIcon(QIcon(str(_PROJECT_ROOT / 'data' / 'icons' / 'start_logo.webp')))

        self.image = ev_img
        self.base_image = ev_img.copy()
        self.conn = child_connection
        self.initUI()
        self.selected_2d_points = []

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        self.image_label = QLabel("2D Image Viewer")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)
        self.image_label.setMouseTracking(True)
        central_widget.setLayout(layout)
        self.setStatusBar(QStatusBar(self))

    def display_image(self):
        """Function to display the image once the image is loaded."""
        # Convert the image for PyQt display
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.base_image = rgb_image.copy()
        self.pixmap = image_to_pixmap(rgb_image)
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setScaledContents(False)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.start_timer()

    def draw_points_on_image(self, points: list, rgb_image: np.ndarray | None = None):
        if rgb_image is not None:
            self.pixmap = image_to_pixmap(rgb_image)
        for idx, point in enumerate(points):
            q_point = QPoint(point[0], point[1])
            self.draw_circle(q_point, idx)

    def mousePressEvent(self, event):
        """Capture mouse click events in the image viewer."""
        if event.button() == Qt.MouseButton.RightButton:
            # Get the relative click position in the QLabel

            label_pos = self.image_label.mapFromGlobal(
                event.globalPosition().toPoint())
            img_x = label_pos.x()
            img_y = label_pos.y()

            img_h, img_w = self.image.shape[:2]
            label_w = self.image_label.width()
            label_h = self.image_label.height()

            # KeepAspectRatio: compute scale and letterbox/pillarbox offsets
            scale = min(label_w / img_w, label_h / img_h)
            offset_x = (label_w - img_w * scale) / 2
            offset_y = (label_h - img_h * scale) / 2

            img_x = int((img_x - offset_x) / scale)
            img_y = int((img_y - offset_y) / scale)

            if self.image_label.pixmap():
                # Ensure the click is within bounds
                if 0 <= img_x < img_w and 0 <= img_y < img_h:
                    self.statusBar().showMessage(f"Selected: ({img_x}, {img_y})")
                    point = (img_x, img_y)
                    self.selected_2d_points.append(point)
                    self.conn.send(point)
                    self.draw_circle(QPoint(*point),
                                     len(self.selected_2d_points)-1)

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
        pen = QPen(QColor("green"))
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

    def undo(self):
        if self.selected_2d_points:
            self.selected_2d_points.pop()
            self.draw_points_on_image(self.selected_2d_points, self.base_image)

    def start_timer(self):
        self.undo_timer = QTimer(self)
        self.undo_timer.timeout.connect(self.undo_poll)
        self.undo_timer.start(100)

    def undo_poll(self):
        while self.conn.poll():
            msg = self.conn.recv()
            if isinstance(msg, tuple):
                command = msg[0]
                if command == "UNDO":
                    self.undo()
                if command == "LOAD":
                    self.selected_2d_points.extend(msg[1])
                    for i, point in enumerate(self.selected_2d_points):
                        self.draw_circle(QPoint(*point), i)
                if command == "UPDATE":
                    self.selected_2d_points = list(msg[1])
                    self.draw_points_on_image(self.selected_2d_points, self.base_image)


class ReprojectionErrorWindow(QDialog):
    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reprojection Error")
        self.data = data
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()

        for k, v in self.data.items():
            form = QFormLayout()
            form.addRow(f"{k}:    ", QLabel(f"{v} px", self))
            layout.addLayout(form)

        self.setLayout(layout)
