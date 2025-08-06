__author__ = "Rahul Jakkamsetty"
__license__ = "MIT"
__doc__ = """

"""

# python imports
import sys
import os
import locale
import multiprocessing as mp
import webbrowser
from threading import Thread

# third-party imports
import comm
import numpy as np
import cv2
import open3d as o3d
import pyvista as pv
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QDialog, QToolBar,
                             QPushButton, QWidget, QLabel, QFileDialog, QHBoxLayout, QStatusBar)
from PyQt6.QtCore import Qt, QPoint, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QIcon



class SecondaryWindow(QMainWindow):
    """Secondary Window allocated to visualized Event Data"""
    def __init__(self, child_connection, ev_img):
        super().__init__()
        self.setWindowTitle("Event Image")
        self.setGeometry(100, 100, ev_img.shape[1], ev_img.shape[0])

        self.image = ev_img
        self.conn = child_connection
        self.initUI()
        self.selected_2d_points = []
        self.image_backups = []
 
        
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
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h,
                         bytes_per_line, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setScaledContents(True)
        self.start_timer()

    def mousePressEvent(self, event):
        """Capture mouse click events in the image viewer."""
        if event.button() == Qt.MouseButton.RightButton:
            # Get the relative click position in the QLabel

            label_pos = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
            img_x = label_pos.x()
            img_y = label_pos.y()

            pixmap_size = self.image_label.pixmap().size()


            label_size = self.image_label.size()

            # Calculate scaling ratio (image might not fill the label fully)
            scale_x = pixmap_size.width() / label_size.width()
            scale_y = pixmap_size.height() / label_size.height()

            img_x = int(img_x * scale_x)
            img_y = int(img_y * scale_y)
            # Calculate the exact pixel coordinates in the image
            if self.image_label.pixmap():
                pixmap_width = self.pixmap.width()
                pixmap_height = self.pixmap.height()

                # Ensure the click is within bounds
                if 0 <= img_x < pixmap_width and 0 <= img_y < pixmap_height:
                    print(f"Exact image pixel coordinates: ({img_x}, {img_y})")
                    point = (img_x, img_y)
                    self.selected_2d_points.append(point)
                    self.conn.send(point)
                    self.image_backups.append(self.pixmap.copy())
                    self.draw_circle(QPoint(*point),
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
        pen = QPen(QColor("green"))
        pen.setWidth(2)
        painter.setPen(pen)
        radius = 3

        brush = QColor("yellow")
        painter.setBrush(brush)
        painter.drawEllipse(position, radius, radius)
        text_offset = position + QPoint(radius+1, -(radius+1))
        painter.drawText(text_offset, f"P{p_index}")
        painter.end()

        self.image_label.setPixmap(self.pixmap)

    def undo(self):
        if self.selected_2d_points:
            self.selected_2d_points.pop()
            img = self.image_backups.pop()
            self.image_label.setPixmap(img)
            self.pixmap = img
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