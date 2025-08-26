__author__ = "Rahul Jakkamsetty"
__license__ = "MIT"
__doc__ = """
Collection of styles for various GUI icons. Developed at DFKI JUL-AUG 2025.
"""
# python imports


# third-party imports
from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QColor, QBrush, QFont

# internal imports




class Switch(QCheckBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setChecked(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(45, 25)  # smaller width

    def paintEvent(self, event):
        # Painter setup
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Track (background)
        track_rect = QRectF(0, 0, self.width(), self.height())
        if self.isChecked():
            track_color = QColor("#4cd964")  # green when ON
        else:
            track_color = QColor("#999")     # gray when OFF
        painter.setBrush(QBrush(track_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(track_rect, self.height()/2, self.height()/2)

        # Knob (circle)
        knob_diameter = self.height() - 6
        if self.isChecked():
            knob_x = self.width() - knob_diameter - 3
        else:
            knob_x = 3
        knob_rect = QRectF(knob_x, 3, knob_diameter, knob_diameter)
        painter.setBrush(QBrush(QColor("#fff")))
        painter.drawEllipse(knob_rect)

        # Text ("ON" / "OFF")
        painter.setPen(QColor("#fff"))
        font = QFont()
        font.setPointSize(7)
        painter.setFont(font)

        if self.isChecked():
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "  ON")
        else:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, "OFF  ")