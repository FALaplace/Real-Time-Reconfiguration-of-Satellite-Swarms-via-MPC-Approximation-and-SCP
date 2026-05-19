import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from Global_Parameters import sat

np.random.seed(41)


class obstacle_motion:
    def __init__(self, n=None, rho=None):
        self.rho = rho
        self.n = n  # non-cooperative target account
        self.w = np.sqrt(sat.mu / sat.a ** 3)  # the average orbital angular velocity
        self.A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [3 * self.w ** 2, 0, 0, 0, 2 * self.w, 0],
            [0, 0, 0, -2 * self.w, 0, 0],
            [0, 0, -self.w ** 2, 0, 0, 0]
        ])
        self.obs_pos = None

    def __init_position(self):
        pass

    def propagate(self, t, obs_pos=None):
        obs_pos_ = np.array(obs_pos) if obs_pos is not None else self.obs_pos
        obs_pos_ = (sp.linalg.expm(self.A * t) @ obs_pos_.T).T
        if obs_pos is None:
            self.obs_pos = obs_pos_
        return obs_pos_

    def collision(self, tf):
        w = self.w
        from numpy import cos, sin

        def funs(vels):
            vx, vy, vz = vels[0], vels[1], vels[2]
            c = vy + 2 * w * x
            eq1 = 2 * vx / w * cos(w * tf) + 2 * (2 * vy / w + 3 * x) * sin(w * tf) - 3 * c * tf + (y - 2 * vx / w)
            eq2 = vx / w * sin(w * tf) - (2 * vy / w + 3 * x) * cos(w * tf) + 2 * c / w
            eq3 = vz / w * sin(w * tf) + z * cos(w * tf)
            return np.array([eq1, eq2, eq3])

        pos = np.random.uniform(low=[-20, -20, -1], high=[20, 20, 1], size=3)
        tf = 0.3 * 2 * np.pi / w if not tf else tf
        x, y, z = pos[0], pos[1], pos[2]
        vel0 = np.array([w, w, w])
        sln = sp.optimize.fsolve(func=funs, x0=vel0, xtol=1e-10, maxfev=1000)
        sln = np.concatenate((pos, sln))
        return sln

    def render(self, T=None):
        if T is None:
            T = 0.6 * np.pi / self.w
        dt = 10
        t = 0
        POS = []
        while t <= T:
            POS.append(self.propagate(dt))
            t += dt
        POS = np.array(POS)
        from visualization import plot_dynamic_obstacles
        plot_dynamic_obstacles(POS[:, :, :3], self.rho)
        plt.show()


class surrounding(obstacle_motion):
    def __init__(self, n=4, r=4.0, rho=0.6):
        super().__init__(n, rho)
        self.r = r  # surrounding radius
        self.obs_pos = self.__init_position()

    def __init_position(self):
        ini_pos = []
        T = 2 * np.pi / self.w
        p0 = np.array([self.r, 0, np.random.uniform(0, 0.5), 0, -2 * self.w * self.r, 0])
        for i in range(self.n):
            ti = T * i / self.n
            expA = sp.linalg.expm(self.A * ti)
            ini_pos.append(expA @ p0)
            pass
        return np.array(ini_pos)


class static(obstacle_motion):
    def __init__(self, n=4, rho=0.6):
        super().__init__(n, rho)
        self.obs_pos = self.__init_position()
        pass

    def __init_position(self):
        return np.random.uniform([-5.0, -5.0, -1.0], [5.0, 5.0, 1.0], size=(self.n, 3))

    def propagate(self, t, obs_pos=None):
        return obs_pos if obs_pos is not None else self.obs_pos


class flyby(obstacle_motion):
    def __init__(self, n=4, rho=0.6):
        super().__init__(n, rho)
        self.obs_pos = self.__init_position()

    def __init_position(self):
        ini_pos = []
        for i in range(self.n):
            x0 = np.random.uniform(-5, 5)
            y0 = np.random.uniform(-5, 5)
            z0 = np.random.uniform(-1, 1)
            ini_pos.append([x0, y0, z0, 0, -3 / 2 * x0 * self.w, 0])
        return np.array(ini_pos)


class OrbitUnit(obstacle_motion):
    def __init__(self, sigma):
        super().__init__(1, 0.6)
        self.sigma = sigma

    def init_position(self, x0, y0, d=0.0):
        kx, bx = -3, -3
        vy0 = (-2 * self.w * x0) * self.sigma + (1 - self.sigma) * (-3 / 2 * self.w * x0)
        if self.sigma == 1.0:
            vx0 = (y0 - d) * self.w / 2
        else:
            vx0 = self.sigma * (kx * y0 * self.w + bx * x0 * self.w)
        z0 = np.random.uniform(-0.5, 0.5)
        self.obs_pos = np.array([x0, y0, z0, vx0, vy0, 0])
        return self.obs_pos.copy()


if __name__ == "__main__":
    sur = surrounding(n=9)
    print(sur.obs_pos)
