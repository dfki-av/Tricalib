import numpy as np
import pytest


def test_project_points_identity_K():
    """With identity K, R, t=0: point [X, Y, Z] projects to [X/Z, Y/Z]."""
    from tricalib.utils.projection import project_points
    points_3d = np.array([[1.0, 2.0, 5.0]])
    R = np.eye(3)
    t = np.zeros(3)
    K = np.eye(3)
    result = project_points(points_3d, R, t, K)
    assert result.shape == (1, 2)
    np.testing.assert_allclose(result[0], [1.0 / 5.0, 2.0 / 5.0], atol=1e-6)


def test_project_points_principal_point():
    """A point directly in front of the camera projects to the principal point."""
    from tricalib.utils.projection import project_points
    fx, fy, cx, cy = 500.0, 400.0, 320.0, 240.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    R = np.eye(3)
    t = np.zeros(3)
    # Point at (0, 0, 1) should map to (cx, cy)
    result = project_points(np.array([[0.0, 0.0, 1.0]]), R, t, K)
    np.testing.assert_allclose(result[0], [cx, cy], atol=1e-6)


def test_project_points_focal_length():
    """Verify focal length scaling: point at (f, 0, f) maps to (cx + f, cy)."""
    from tricalib.utils.projection import project_points
    fx, fy, cx, cy = 500.0, 400.0, 320.0, 240.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    R = np.eye(3)
    t = np.zeros(3)
    result = project_points(np.array([[fx, 0.0, fx]]), R, t, K)
    np.testing.assert_allclose(result[0], [cx + fx, cy], atol=1e-6)


def test_project_points_multiple():
    """Batch projection returns one row per input point."""
    from tricalib.utils.projection import project_points
    K = np.eye(3)
    R = np.eye(3)
    t = np.zeros(3)
    points = np.array([[1.0, 0.0, 2.0],
                        [0.0, 1.0, 4.0],
                        [3.0, 6.0, 3.0]])
    result = project_points(points, R, t, K)
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result[0], [0.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(result[1], [0.0, 0.25], atol=1e-6)
    np.testing.assert_allclose(result[2], [1.0, 2.0], atol=1e-6)
