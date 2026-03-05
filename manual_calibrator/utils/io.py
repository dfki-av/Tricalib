__author__ = "Rahul Jakkamsetty"
__license__ = "CC BY-NC-SA 4.0"
__doc__ = """
Few io utility functions. 
Developed at DFKI in DEC-JAN 2024-25.
"""

# python imports
import json
from typing import Any

# third-party imports
import yaml
import numpy as np
import open3d as o3d
from PyQt6.QtGui import QPixmap, QImage, QPainter, QIcon, QFont
from PyQt6.QtCore import Qt
import imageio.v2 as imageio



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
    if not file_path.endswith('.json'):
        file_path += '.json'
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_point_cloud(file_path: str) -> np.ndarray:
    """
    loads the PCD point cloud from the disk.
    """
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

def read_image(file_path: str) -> np.ndarray:
    """
    loads the image from the disk.
    """
    img = imageio.imread(file_path)
    return img

def save_image(file_path: str, img_data: np.ndarray) -> None:
    """
    save the image to the disk.
    """
    imageio.imsave(file_path, img_data)
    

def read_point_cloud(file_path:str) -> np.ndarray:
    """
    loads the PCD point cloud with intensities from the disk.
    """
    pcd = o3d.t.io.read_point_cloud(
                file_path, format='auto')
    pcd_points = pcd.point.positions.numpy()
    pcd_intensity = pcd.point.intensity.numpy().reshape(-1, 1)
    return np.hstack((pcd_points, pcd_intensity))

def load_yaml(file_path: str):
    """
    loads the data from YAML file.
    """
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def ucode_icon(unicode: str | list[str]):
    """
    Creates a QIcon from unicode character(s).
    If multiple unicode strings are passed, merges them into a single icon.
    """
    unicode_list = unicode if isinstance(unicode, list) else [unicode]
    
    img = QImage(100, 100, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)

    painter = QPainter(img)
    painter.setFont(QFont("Arial", 48))
    painter.setPen(Qt.GlobalColor.black)
    
    text = "".join(unicode_list)
    painter.drawText(img.rect(), Qt.AlignmentFlag.AlignCenter, text)
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


def serialize_dict(data: dict) -> dict:
    """
    serializes the data with numpy arrays to save to JSON file.
    """
    for k in data:
        if hasattr(data[k], 'tolist'):
            data[k] = data[k].tolist()
    return data
