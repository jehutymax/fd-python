import numpy as np
import matplotlib.pyplot as plt
import math


def solver(I, w, dt, T):
    """
    Solve u'' + w**2 * u = 0 for t in (0, T], u(0)=I and u'(0) = 0
    using a central finite difference method with time step dt.
    :param I: initial condition u(0)=I
    :param w: (omega) in the ODE
    :param dt: time step
    :param T: time limit - solve for t in (0, T]
    :return: u and t arrays
    """
    dt = float(dt)
    num_points = int(round(T / dt))
    u = np.zeros(num_points + 1)
    t = np.linspace(0, num_points * dt, num_points + 1)

    u[0] = I
    u[1] = u[0] - 0.5 * u[0] * dt * dt * w * w

    for n in range(1, num_points):
        u[n + 1] = - u[n-1] + 2 * u[n] - dt * dt * w * w * u[n]

    return u, t


def u_exact(t, I, w):
    return I * np.cos(w * t)


def visualize(u, t, I, w):
    plt.plot(t, u, 'r--o')
    t_fine = np.linspace(0, t[-1], 1001)
    u_correct = u_exact(t_fine, I, w)
    plt.plot(t_fine, u_correct, 'b-')
    plt.legend(['numerical', 'exact'], loc="upper left")
    plt.xlabel("t")
    plt.ylabel("u")
    dt = t[1] - t[0]
    plt.title("dt = %g" % dt)
    u_min = 1.2 * u.min()
    u_max = -u_min
    plt.axis([t[0], t[-1], u_min, u_max])
    plt.savefig("plot1.png")
    # plt.show()


def test_first_three_steps():
    from math import pi
    I = 1; w = 2*pi; dt = 0.1; T = 1;
    expected = np.array([1.00, 0.802607911978213, 0.288358920740053])
    u, t = solver(I, w, dt, T)
    diff = np.abs(expected - u[:3]).max()
    tol = 1e-14
    assert diff < tol


I = 1
w = 2 * math.pi
dt = 0.05
num_periods = 5
P = math.pi * 2/w
T = P * num_periods
u, t = solver(I, w, dt, T)
visualize(u, t, I, w)
