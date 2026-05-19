"""
建立J2摄动下的航天器运动方程和相对运动方程
"""
import numpy as np
from numpy.linalg import norm
from numpy import cos, sin, tan
from scipy.linalg import expm
from scipy.optimize import approx_fprime
import casadi as ca

r2d = lambda x: x * 180 / np.pi
Re = 6371.0
mu = 398600.4418
J2 = 1.08264 * 10 ** (-3)

k_J2 = 3 * J2 * mu * Re ** 2 / 2
ka = 0.0


class RelativeMotionModel_J2:
    """
    The center satellite state is denoted as sc; the j-th follow satellite state is denoted as sj
    sc = [r, vx, h, the, inc, RAAN]
    sj = [x, y, z, vx, vy, vz]
    """

    def __init__(self, chief_sat_elems: list):
        self.csat_state = np.array(chief_sat_elems)
        self.c0 = self.csat_state.copy()
        self.csat_state_ECI = None
        self.B = np.concatenate((np.zeros((3, 3)), np.eye(3)), axis=0)
        self.G = np.concatenate((np.eye(3), np.zeros((3, 3))), axis=1)
        self.__h = 10

    def set_sample_time(self, h: int | float):
        self.__h = h

    def get_sample_time(self):
        return self.__h

    def derivation_csat(self):
        r, vx, h, the, i, RAAN = self.csat_state
        dr = vx
        dvx = -mu / r ** 2 + h ** 2 / r ** 3 - k_J2 * (1 - 3 * sin(i) ** 2 * sin(the) ** 2) / r ** 4
        dh = -k_J2 * (sin(i) ** 2 * sin(2 * the)) / r ** 3
        dthe = h / r ** 2 + 2 * k_J2 * (cos(i) ** 2 * sin(the) ** 2) / (h * r ** 3)
        dinc = -k_J2 * (sin(2 * i) * sin(2 * the)) / (2 * h * r ** 3)
        dRAAN = -2 * k_J2 * cos(i) * sin(the) ** 2 / (h * r ** 3)
        return np.array([dr, dvx, dh, dthe, dinc, dRAAN])

    @staticmethod
    def ds_fsat(s, u, args: np.ndarray):
        r, vx, h, the, i, raan = args
        wx = -k_J2 * (sin(2 * i) * sin(the)) / (h * r ** 3)
        wz = h / r ** 2
        ax = (-k_J2 * (sin(2 * i) * cos(the)) / r ** 5 + 3 * vx * k_J2 * (sin(2 * i) * sin(the)) / (r ** 4 * h) -
              8 * k_J2 ** 2 * sin(i) ** 3 * cos(i) * sin(the) ** 2 * cos(the) / (r ** 6 * h ** 2))
        az = -2 * h * vx / r ** 3 - k_J2 * (sin(i) ** 2 * sin(2 * the)) / r ** 5
        eta = mu / r ** 3 + k_J2 / r ** 5 - 5 * k_J2 * sin(i) ** 2 * sin(the) ** 2 / r ** 5
        kexi = 2 * k_J2 * (sin(i) * sin(the)) / r ** 4
        x, y, z, vx, vy, vz = s[0], s[1], s[2], s[3], s[4], s[5]
        rf = np.sqrt((r + x) ** 2 + y ** 2 + z ** 2)
        rfz = (r + x) * sin(i) * sin(the) + y * sin(i) * cos(the) + z * cos(i)
        etaf = mu / rf ** 3 + k_J2 / rf ** 5 - 5 * k_J2 * rfz ** 2 / rf ** 7
        kexif = 2 * k_J2 * rfz / rf ** 5
        dx, dy, dz = vx, vy, vz
        dvx = (2 * vy * wz - x * (etaf - wz ** 2) + y * az - z * wx * wz -
               (kexif - kexi) * sin(i) * sin(the) - r * (etaf - eta) + u[0])
        dvy = (-2 * vx * wz + 2 * vz * wx - x * az - y * (etaf - wz ** 2 - wx ** 2) + z * ax -
               (kexif - kexi) * sin(i) * cos(the) + u[1])
        dvz = -2 * vy * wx - x * wx * wz - y * ax - z * (etaf - wx ** 2) - (kexif - kexi) * cos(i) + u[2]
        return [dx, dy, dz, dvx, dvy, dvz]

    def discrete_dynamic(self, s, u):
        k1 = ca.vertcat(*self.ds_fsat(s, u, self.csat_state))
        k2 = ca.vertcat(*self.ds_fsat(s + 0.5 * self.__h * k1, u, self.csat_state))
        k3 = ca.vertcat(*self.ds_fsat(s + 0.5 * self.__h * k2, u, self.csat_state))
        k4 = ca.vertcat(*self.ds_fsat(s + self.__h * k3, u, self.csat_state))
        return s + self.__h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def update(self, s, u):
        k1 = np.array(self.ds_fsat(s, u, self.csat_state))
        k2 = np.array(self.ds_fsat(s + 0.5 * self.__h * k1, u, self.csat_state))
        k3 = np.array(self.ds_fsat(s + 0.5 * self.__h * k2, u, self.csat_state))
        k4 = np.array(self.ds_fsat(s + self.__h * k3, u, self.csat_state))
        return s + self.__h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def Jacboi_matrix(self, x, args=None):
        if args is None:
            args = [0, 0, 0], self.csat_state
        jac = approx_fprime(x, self.ds_fsat, np.array([1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9]), [0, 0, 0], self.csat_state)
        return jac

    def local2ECI(self):
        inc = self.csat_state[4]
        the = self.csat_state[3]
        RAAN = self.csat_state[5]
        C1 = np.array([[cos(RAAN), sin(RAAN), 0.0],
                       [-sin(RAAN), cos(RAAN), 0.0],
                       [0.0, 0.0, 1.0]])
        C2 = np.array([[1, 0, 0],
                       [0, cos(inc), sin(inc)],
                       [0.0, -sin(inc), cos(inc)]])
        C3 = np.array([[cos(the), sin(the), 0.0],
                       [-sin(the), cos(the), 0.0],
                       [0.0, 0.0, 1.0]])
        C = C3 @ C2 @ C1
        C_inv = np.linalg.inv(C)
        csat_r = C_inv @ np.array([self.csat_state[0], 0.0, 0.0])
        return csat_r
