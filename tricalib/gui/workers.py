__author__ = "Rahul Jakkamsetty"
__license__ = "CC BY-NC-SA 4.0"
__doc__ = """
TriCalib - workers
=======================================
Contains the worker functions the launch separate windows.
Developed at DFKI (German Research Center for AI), March 2026.
"""
# python imports
import os
import sys

# third-party imports
import pyvista as pv
from PyQt6.QtWidgets import QApplication

# internal imports
from tricalib.gui.style import DARK_STYLESHEET, LIGHT_STYLESHEET
from tricalib.gui.secgui import SecondaryWindow




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
