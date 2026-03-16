import numpy as np
import pytest


def test_parameters_to_matrices_identity():
    """Identity quaternions and zero translations should give identity rotation blocks."""
    from tricalib.optim.optimizer import parameters_to_matrices
    # scipy quaternion convention: [x, y, z, w]; identity = [0, 0, 0, 1]
    params = np.zeros(14)
    params[3] = 1.0   # lidar2rgb w
    params[10] = 1.0  # lidar2evt w
    result = parameters_to_matrices(params)
    assert set(result.keys()) >= {'T_lidar_to_rgb', 'T_lidar_to_evt', 'T_rgb_to_evt'}
    np.testing.assert_allclose(result['T_lidar_to_rgb'][:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(result['T_lidar_to_evt'][:3, :3], np.eye(3), atol=1e-6)
    # Zero translation
    np.testing.assert_allclose(result['T_lidar_to_rgb'][:3, 3], np.zeros(3), atol=1e-6)


def test_parameters_to_matrices_shape():
    """Output matrices must be 4×4."""
    from tricalib.optim.optimizer import parameters_to_matrices
    params = np.zeros(21)
    params[3] = 1.0; params[10] = 1.0; params[17] = 1.0
    result = parameters_to_matrices(params)
    for key in ('T_lidar_to_rgb', 'T_lidar_to_evt', 'T_rgb_to_evt'):
        assert result[key].shape == (4, 4), f"{key} has wrong shape"


def test_reprojection_error_raises_on_wrong_K_rgb_shape():
    from tricalib.optim.optimizer import reprojection_error
    params = np.zeros(21)
    params[3] = 1.0; params[10] = 1.0; params[17] = 1.0
    K_bad = np.eye(4)   # wrong shape
    K_good = np.eye(3)
    rect = dict(rgb=np.eye(3), event=np.eye(3))
    with pytest.raises(ValueError, match="K_rgb"):
        reprojection_error(params, [], [], [], K_bad, K_good, rect_matrics=rect)


def test_reprojection_error_raises_on_wrong_K_ev_shape():
    from tricalib.optim.optimizer import reprojection_error
    params = np.zeros(21)
    params[3] = 1.0; params[10] = 1.0; params[17] = 1.0
    K_good = np.eye(3)
    K_bad = np.eye(4)
    rect = dict(rgb=np.eye(3), event=np.eye(3))
    with pytest.raises(ValueError, match="K_ev"):
        reprojection_error(params, [], [], [], K_good, K_bad, rect_matrics=rect)
