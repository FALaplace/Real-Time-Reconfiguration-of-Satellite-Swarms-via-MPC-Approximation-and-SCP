import numpy as np
import math
from MotionModel import RelativeMotionModel_J2
from sklearn.gaussian_process.kernels import Matern

r2d = lambda x: x * 180 / math.pi
d2r = lambda x: x * math.pi / 180


class op:
    Q = np.diag([6, 6, 6, 100, 100, 100]) ** 2
    R = np.diag([5e6, 5e6, 5e6])
    dim_x = 6
    dim_u = 3
    pos_max = 20
    vel_max = 0.02
    x_max = [pos_max] * 3 + [vel_max] * 3
    u_max = 3e-5
    u_max_list = [u_max] * dim_u
    normalX = lambda x: np.clip((x + np.array(op.x_max)) / (2 * np.array(op.x_max)), 0, 1)
    normalU = lambda x: np.clip((x + op.u_max) / (2 * op.u_max), 0, 1)
    inverseU = lambda x: 2 * x * op.u_max - op.u_max
    "SCP global parameters"
    tau = 2.0  # trust region constraint: X[k] - Xr[k] <= tau
    pun = 1e5  # collision avoidance (soft constraints) punishment: cost = pun * (ro - ||x - po||)
    rho = 0.7  # collision avoidance (asymptotic constraint): ||x - po|| > ro * rho


class sat:
    Re = 6371.0
    mu = 398600.4418
    J2 = 1.08264 * 10 ** (-3)
    a, e, inc, w, RAAN, TA = Re + 900, 0.0, d2r(60), d2r(45), d2r(50.2), d2r(30)
    r = a * (1 - e ** 2) / (1 + e * np.cos(TA))
    h = np.sqrt(a * (1 - e ** 2) * mu)
    vx = np.sqrt(np.round(2 * mu / r - mu / a - h ** 2 / r ** 2, 10))
    the = w + TA
    omega = (mu / a ** 3) ** (1 / 2)
    rm = RelativeMotionModel_J2([r, vx, h, the, inc, RAAN])
    x_tar = np.zeros(op.dim_x)
    u_tar = -np.array(rm.ds_fsat(x_tar, [0] * 3, rm.csat_state)[3:])
    safe_dis = 0.2  # safe distance between satellites
    view_rge = 3.0  # view range of each satellites


class gd:
    Mp = 4
    Mv = 4
    ls = 0.25
    c_ls = 0.8
    nu = 3 / 2
    kernel = Matern(length_scale=ls * c_ls, nu=nu, length_scale_bounds='fixed')




