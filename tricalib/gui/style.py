__author__ = "Rahul Jakkamsetty"
__license__ = "CC BY-NC-SA 4.0"
__doc__ = """
TriCalib - GUI Style Components
==========================================
Provides custom Qt widget subclasses for a polished toolbar appearance:

  - :class:`Switch` — a toggle switch that renders as a rounded pill with a
    sliding knob and ON/OFF text labels, used for the Stereo Rectification
    and Auto Axis Alignment toolbar controls.

Developed at DFKI (German Research Center for AI), July – August 2025.
"""
# python imports


# third-party imports
from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtCore import Qt, QRectF, pyqtProperty, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPainter, QColor, QBrush, QPixmap, QIcon

# internal imports


# ── Icon tint colors per theme ────────────────────────────────────────────────
ICON_COLOR_DARK = "#cdd9e5"
ICON_COLOR_LIGHT = "#3c4553"

# ── Dark theme — charcoal + teal accent ───────────────────────────────────────
DARK_STYLESHEET = """
/* ─── Base ──────────────────────────────────────────────────────────── */
QWidget {
    background-color: #1c2128;
    color: #cdd9e5;
    selection-background-color: #1d6b6b;
    selection-color: #ffffff;
}

/* ─── Main Window ───────────────────────────────────────────────────── */
QMainWindow {
    background-color: #1c2128;
}

/* ─── Menu Bar ──────────────────────────────────────────────────────── */
QMenuBar {
    background-color: #22272e;
    color: #cdd9e5;
    border-bottom: 1px solid #373e47;
    padding: 1px 4px;
    spacing: 0px;
}
QMenuBar::item {
    padding: 5px 10px;
    border-radius: 3px;
    background: transparent;
}
QMenuBar::item:selected {
    background-color: #2d333b;
    color: #cdd9e5;
}
QMenuBar::item:pressed {
    background-color: #1d6b6b;
    color: #ffffff;
}

/* ─── Drop-down Menus ───────────────────────────────────────────────── */
QMenu {
    background-color: #22272e;
    border: 1px solid #444c56;
    padding: 3px 0px;
}
QMenu::item {
    padding: 5px 28px 5px 14px;
    background: transparent;
}
QMenu::item:selected {
    background-color: #2d333b;
    color: #4ecdc4;
}
QMenu::item:disabled {
    color: #4d5566;
}
QMenu::separator {
    height: 1px;
    background-color: #373e47;
    margin: 3px 6px;
}

/* ─── Toolbars ──────────────────────────────────────────────────────── */
QToolBar {
    background-color: #22272e;
    border: none;
    border-right: 1px solid #373e47;
    padding: 4px 2px;
    spacing: 2px;
}
QToolBar[orientation="2"] {
    border-right: none;
    border-bottom: 1px solid #373e47;
}
QToolBar::separator {
    background-color: #373e47;
    width: 1px;
    margin: 3px 2px;
}
QToolButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 5px;
    color: #cdd9e5;
    min-width: 24px;
    min-height: 24px;
}
QToolButton:hover {
    background-color: #2d333b;
    border-color: #444c56;
}
QToolButton:pressed {
    background-color: #1d6b6b;
    border-color: #2da8a8;
}

/* ─── Status Bar ────────────────────────────────────────────────────── */
QStatusBar {
    background-color: #2da8a8;
    color: #ffffff;
    border: none;
}
QStatusBar::item {
    border: none;
}
QStatusBar QLabel {
    color: #ffffff;
    background-color: transparent;
    padding: 0px 4px;
}

/* ─── Labels ────────────────────────────────────────────────────────── */
QLabel {
    background-color: transparent;
    color: #cdd9e5;
}

/* ─── Dialogs ───────────────────────────────────────────────────────── */
QDialog {
    background-color: #1c2128;
}
QMessageBox {
    background-color: #22272e;
}
QMessageBox QLabel {
    color: #cdd9e5;
}

/* ─── Push Buttons ──────────────────────────────────────────────────── */
QPushButton {
    background-color: #2d333b;
    border: 1px solid #444c56;
    border-radius: 4px;
    padding: 5px 14px;
    color: #cdd9e5;
    min-width: 70px;
}
QPushButton:hover {
    background-color: #373e47;
    border-color: #2da8a8;
    color: #ffffff;
}
QPushButton:pressed {
    background-color: #1d6b6b;
    border-color: #2da8a8;
    color: #ffffff;
}
QPushButton:focus {
    border-color: #2da8a8;
    outline: none;
}
QPushButton:default {
    border-color: #2da8a8;
}

/* ─── Line Edits ────────────────────────────────────────────────────── */
QLineEdit {
    background-color: #2d333b;
    border: 1px solid #444c56;
    border-radius: 3px;
    padding: 4px 8px;
    color: #cdd9e5;
    selection-background-color: #1d6b6b;
}
QLineEdit:focus {
    border-color: #2da8a8;
    outline: none;
}

/* ─── List & Tree Views (File Dialog) ──────────────────────────────── */
QTreeView, QListView {
    background-color: #22272e;
    border: 1px solid #373e47;
    color: #cdd9e5;
    alternate-background-color: #1c2128;
}
QTreeView::item:selected, QListView::item:selected {
    background-color: #1d6b6b;
    color: #ffffff;
}
QTreeView::item:hover, QListView::item:hover {
    background-color: #2d333b;
}
QHeaderView::section {
    background-color: #22272e;
    color: #cdd9e5;
    border: 1px solid #373e47;
    padding: 4px 6px;
}

/* ─── Scroll Bars ───────────────────────────────────────────────────── */
QScrollBar:vertical {
    background-color: #1c2128;
    width: 12px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background-color: #444c56;
    min-height: 20px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar::handle:vertical:hover {
    background-color: #57606a;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background-color: #1c2128;
    height: 12px;
    margin: 0px;
}
QScrollBar::handle:horizontal {
    background-color: #444c56;
    min-width: 20px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #57606a;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ─── Combo Boxes ───────────────────────────────────────────────────── */
QComboBox {
    background-color: #2d333b;
    border: 1px solid #444c56;
    border-radius: 3px;
    padding: 4px 8px;
    color: #cdd9e5;
    min-width: 80px;
}
QComboBox:hover {
    border-color: #2da8a8;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #22272e;
    border: 1px solid #444c56;
    color: #cdd9e5;
    selection-background-color: #1d6b6b;
}
"""

# ── Light theme — off-white + teal accent ─────────────────────────────────────
LIGHT_STYLESHEET = """
/* ─── Base ──────────────────────────────────────────────────────────── */
QWidget {
    background-color: #f6f8fa;
    color: #1f2328;
    selection-background-color: #b8e8e8;
    selection-color: #1f2328;
}

/* ─── Main Window ───────────────────────────────────────────────────── */
QMainWindow {
    background-color: #f6f8fa;
}

/* ─── Menu Bar ──────────────────────────────────────────────────────── */
QMenuBar {
    background-color: #eaeef2;
    color: #1f2328;
    border-bottom: 1px solid #d0d7de;
    padding: 1px 4px;
    spacing: 0px;
}
QMenuBar::item {
    padding: 5px 10px;
    border-radius: 3px;
    background: transparent;
}
QMenuBar::item:selected {
    background-color: #dde3ea;
    color: #1f2328;
}
QMenuBar::item:pressed {
    background-color: #b8e8e8;
    color: #1a7070;
}

/* ─── Drop-down Menus ───────────────────────────────────────────────── */
QMenu {
    background-color: #ffffff;
    border: 1px solid #c8d1db;
    padding: 3px 0px;
}
QMenu::item {
    padding: 5px 28px 5px 14px;
    background: transparent;
}
QMenu::item:selected {
    background-color: #ddf4f4;
    color: #1a7070;
}
QMenu::item:disabled {
    color: #aab0bb;
}
QMenu::separator {
    height: 1px;
    background-color: #d0d7de;
    margin: 3px 6px;
}

/* ─── Toolbars ──────────────────────────────────────────────────────── */
QToolBar {
    background-color: #eaeef2;
    border: none;
    border-right: 1px solid #d0d7de;
    padding: 4px 2px;
    spacing: 2px;
}
QToolBar[orientation="2"] {
    border-right: none;
    border-bottom: 1px solid #d0d7de;
}
QToolBar::separator {
    background-color: #d0d7de;
    width: 1px;
    margin: 3px 2px;
}
QToolButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 5px;
    color: #1f2328;
    min-width: 24px;
    min-height: 24px;
}
QToolButton:hover {
    background-color: #dde3ea;
    border-color: #c8d1db;
}
QToolButton:pressed {
    background-color: #b8e8e8;
    border-color: #2da8a8;
}

/* ─── Status Bar ────────────────────────────────────────────────────── */
QStatusBar {
    background-color: #2da8a8;
    color: #ffffff;
    border: none;
}
QStatusBar::item {
    border: none;
}
QStatusBar QLabel {
    color: #ffffff;
    background-color: transparent;
    padding: 0px 4px;
}

/* ─── Labels ────────────────────────────────────────────────────────── */
QLabel {
    background-color: transparent;
    color: #1f2328;
}

/* ─── Dialogs ───────────────────────────────────────────────────────── */
QDialog {
    background-color: #f6f8fa;
}
QMessageBox {
    background-color: #ffffff;
}
QMessageBox QLabel {
    color: #1f2328;
}

/* ─── Push Buttons ──────────────────────────────────────────────────── */
QPushButton {
    background-color: #eaeef2;
    border: 1px solid #c8d1db;
    border-radius: 4px;
    padding: 5px 14px;
    color: #1f2328;
    min-width: 70px;
}
QPushButton:hover {
    background-color: #ddf4f4;
    border-color: #2da8a8;
    color: #1a7070;
}
QPushButton:pressed {
    background-color: #b8e8e8;
    border-color: #2da8a8;
    color: #1a7070;
}
QPushButton:focus {
    border-color: #2da8a8;
    outline: none;
}
QPushButton:default {
    border-color: #2da8a8;
}

/* ─── Line Edits ────────────────────────────────────────────────────── */
QLineEdit {
    background-color: #ffffff;
    border: 1px solid #c8d1db;
    border-radius: 3px;
    padding: 4px 8px;
    color: #1f2328;
    selection-background-color: #b8e8e8;
}
QLineEdit:focus {
    border-color: #2da8a8;
    outline: none;
}

/* ─── List & Tree Views (File Dialog) ──────────────────────────────── */
QTreeView, QListView {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    color: #1f2328;
    alternate-background-color: #f6f8fa;
}
QTreeView::item:selected, QListView::item:selected {
    background-color: #b8e8e8;
    color: #1a7070;
}
QTreeView::item:hover, QListView::item:hover {
    background-color: #eaeef2;
}
QHeaderView::section {
    background-color: #eaeef2;
    color: #1f2328;
    border: 1px solid #d0d7de;
    padding: 4px 6px;
}

/* ─── Scroll Bars ───────────────────────────────────────────────────── */
QScrollBar:vertical {
    background-color: #f6f8fa;
    width: 12px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background-color: #c8d1db;
    min-height: 20px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar::handle:vertical:hover {
    background-color: #aab0bb;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background-color: #f6f8fa;
    height: 12px;
    margin: 0px;
}
QScrollBar::handle:horizontal {
    background-color: #c8d1db;
    min-width: 20px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #aab0bb;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ─── Combo Boxes ───────────────────────────────────────────────────── */
QComboBox {
    background-color: #ffffff;
    border: 1px solid #c8d1db;
    border-radius: 3px;
    padding: 4px 8px;
    color: #1f2328;
    min-width: 80px;
}
QComboBox:hover {
    border-color: #2da8a8;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #c8d1db;
    color: #1f2328;
    selection-background-color: #b8e8e8;
}
"""


def themed_icon(path: str, color: str = "#cccccc") -> QIcon:
    """Load an SVG/image icon and recolor it to *color* for dark-mode toolbars.

    Works by drawing the source pixmap onto a transparent canvas then using
    CompositionMode_SourceIn to flood-fill only the non-transparent pixels with
    the requested color, leaving the original alpha channel intact.
    """
    pixmap = QPixmap(path)
    if pixmap.isNull():
        return QIcon(path)

    colored = QPixmap(pixmap.size())
    colored.fill(Qt.GlobalColor.transparent)

    painter = QPainter(colored)
    painter.drawPixmap(0, 0, pixmap)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(colored.rect(), QColor(color))
    painter.end()

    return QIcon(colored)


class Switch(QCheckBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setChecked(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(36, 20)
        self._knob_x = 2.0
        self._anim = QPropertyAnimation(self, b"knob_x", self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.stateChanged.connect(self._start_animation)

    def _knob_target(self):
        diameter = self.height() - 4
        return float(self.width() - diameter - 2) if self.isChecked() else 2.0

    def _start_animation(self):
        self._anim.stop()
        self._anim.setStartValue(self._knob_x)
        self._anim.setEndValue(self._knob_target())
        self._anim.start()

    @pyqtProperty(float)
    def knob_x(self):
        return self._knob_x

    @knob_x.setter
    def knob_x(self, value):
        self._knob_x = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        r = self.height() / 2
        track_rect = QRectF(0, 0, self.width(), self.height())
        track_color = QColor("#2da8a8") if self.isChecked() else QColor("#666")
        painter.setBrush(QBrush(track_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(track_rect, r, r)

        diameter = self.height() - 4
        knob_rect = QRectF(self._knob_x, 2, diameter, diameter)
        painter.setBrush(QBrush(QColor("#fff")))
        painter.drawEllipse(knob_rect)