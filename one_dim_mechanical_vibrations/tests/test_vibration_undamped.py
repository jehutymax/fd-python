from one_dim_mechanical_vibrations.vibration_undamped import *


def test_first_three_steps():
    from math import pi
    I = 1; w = 2*pi; dt = 0.1; T = 1;
    expected = np.array([1.00, 0.802607911978213, 0.288358920740053])
    u, t = solver(I, w, dt, T)
    diff = np.abs(expected - u[:3]).max()
    tol = 1e-14
    assert diff < tol
