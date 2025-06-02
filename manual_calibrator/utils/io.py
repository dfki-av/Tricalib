__author__ = "Rahul Jakkamsetty"
__license__ = "MIT"
__doc__ = """
Few io utility functions. 
Developed at DFKI in DEC-JAN 2024-25.
"""

# python imports
import json
from typing import Any

# third-party imports
import numpy as np
import open3d as o3d
from PyQt5.QtGui import QPixmap, QImage, QPainter, QIcon, QFont
from PyQt5.QtCore import Qt

def load_json(file_path: str) -> Any:
    """
    loads the data from JSON file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def write_json(file_path: str, data: Any) -> None:
    """
    writes the data to a JSON file on disk.
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_point_cloud(file_path: str) -> np.array:
    """
    loads the PCD point cloud from the disk.
    """
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)


def ucode_icon(unicode:str):
    img = QImage(100, 100, QImage.Format_ARGB32)
    img.fill(Qt.transparent)

    painter = QPainter(img)
    painter.setFont(QFont("Arial", 48))
    painter.setPen(Qt.black)
    painter.drawText(img.rect(), Qt.AlignCenter, unicode)
    painter.end()

    pixmap = QPixmap.fromImage(img)
    return QIcon(pixmap)

def fxfycxcy_to_matrix(_4e_fmt):
    if len(_4e_fmt) == 4:
        matrix = np.eye(3)
        matrix[0][0] = _4e_fmt[0]
        matrix[1][1] = _4e_fmt[1]
        matrix[:2, 2] = _4e_fmt[2:]
        return matrix
    return _4e_fmt
