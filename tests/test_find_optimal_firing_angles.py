import numpy as np
from find_optimal_firing_angles import find_optimal_firing_angles


def test_stationary_target():
    shooter = np.array([0, 0, 0])
    target = np.array([1000, 0, 0])
    v_t = np.array([0, 0, 0])
    v_p = 1000.0
    elev, azim, t, hit, min_dist = find_optimal_firing_angles(
        shooter, target, v_t, v_p, "tpt"
    )
    assert np.isclose(azim, 0, atol=1e-2)
    assert np.isclose(elev, 0, atol=1e-2)
    # Accept t in a realistic range due to drag and gravity
    assert 1.1 < t < 1.2
    assert np.allclose(hit, [1000, 0, 0], atol=5.0)
    assert min_dist < 5.0


def test_stationary_target_simple():
    """Test with very simple case first"""
    shooter = np.array([0, 0, 0])
    target = np.array([500, 0, 0])  # Closer target
    v_t = np.array([0, 0, 0])
    v_p = 1000.0
    elev, azim, t, hit, min_dist = find_optimal_firing_angles(
        shooter, target, v_t, v_p, "tpt", max_time=10.0
    )
    print(f"Result: dist={min_dist:.3f}m, elev={np.degrees(elev):.2f}°")


def test_moving_target_simple():
    shooter = np.array([0, 0, 0])
    target = np.array([100, 0, 0])  # Place target 100m ahead
    v_t = np.array([100, 0, 0])
    v_p = 200.0
    elev, azim, t, hit, min_dist = find_optimal_firing_angles(
        shooter, target, v_t, v_p, "tpt"
    )
    assert t > 0
    assert hit[0] > 0
    assert np.isclose(azim, 0, atol=0.2)
    assert np.isclose(elev, 0, atol=0.2)
    assert min_dist < 5.0


def test_moving_target_3d():
    shooter = np.array([0, 0, 0])
    target = np.array([100, 100, 100])  # Place target 100m away in all axes
    v_t = np.array([100, 100, 100])
    v_p = 300.0
    elev, azim, t, hit, min_dist = find_optimal_firing_angles(
        shooter, target, v_t, v_p, "tpt"
    )
    assert t > 0
    assert hit[0] > 0 and hit[1] > 0 and hit[2] > 0
    assert np.isclose(azim, np.pi / 4, atol=0.2)
    assert np.isclose(elev, np.arctan2(1, np.sqrt(2)), atol=0.2)
    assert min_dist < 5.0
