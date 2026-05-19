import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from typing import List


def plot_satellite_trajectory(xi, fig: plt.Figure = None, ax: Axes3D = None):
    if ax is None:
        fig: Figure = plt.figure(figsize=(8, 8))
        ax: Axes3D = fig.add_subplot(projection="3d")
        ax.set_xlabel("x(km)")
        ax.set_ylabel("y(km)")
        ax.set_zlabel("z(km)")
        ax.set_xlim3d(-10, 10)
        ax.set_ylim3d(-10, 10)
        ax.set_zlim3d(-5, 5)
    ax.plot(xi[:, 0], xi[:, 1], xi[:, 2], linewidth=2, alpha=0.7)
    # ax.scatter(xi[0, 0], xi[0, 1], xi[0, 2], s=75, marker='o', edgecolors='black', linewidths=1, alpha=1, zorder=5)
    return fig, ax


def plot_trajectory(s, tar=None, ifsave=False):
    if tar is None:
        tar = [0, 0, 0]
    f3: Figure = plt.figure(figsize=(8, 8))
    ax: Axes3D = f3.add_subplot(projection="3d")
    ax.scatter3D(tar[0], tar[1], tar[2], label="Target Spacecraft", marker="*")
    ax.scatter3D(s[0, 0], s[0, 1], s[0, 2], label="Initial State", marker="x")
    ax.plot3D(s[:, 0], s[:, 1], s[:, 2], label="real track")

    lower = np.min(s[:, [0, 1, 2]]) * 1.1
    upper = np.max(s[:, [0, 1, 2]]) * 1.1

    ax.set_xlabel("x(km)")
    ax.set_ylabel("y(km)")
    ax.set_zlabel("z(km)")
    ax.set_xlim3d(lower, upper)
    ax.set_ylim3d(lower, upper)
    ax.set_zlim3d(lower, upper)
    ax.legend()
    if ifsave:
        f3.savefig("figure/track.png")
    else:
        plt.show()
        pass


def plot_trajectory_with_obstacles(X_value, Xr, obstacles):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- Plot actual trajectory ---
    ax.plot(
        X_value[:, 0], X_value[:, 1], X_value[:, 2],
        label="Actual Trajectory (X_value)",
        linewidth=2.5
    )

    if Xr is not None:
        # --- Plot reference trajectory ---
        ax.plot(
            Xr[:, 0], Xr[:, 1], Xr[:, 2],
            label="Reference Trajectory (Xr)",
            linestyle='--',
            linewidth=2.0
        )

    # --- Plot obstacles ---
    for center, radius in obstacles:
        cx, cy, cz = center

        # 球面参数方程
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        xs = cx + radius * np.outer(np.cos(u), np.sin(v))
        ys = cy + radius * np.outer(np.sin(u), np.sin(v))
        zs = cz + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(
            xs, ys, zs,
            rstride=1, cstride=1,
            color='r', alpha=0.3,
            linewidth=0
        )

    # Labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Trajectory with Collision Obstacles")

    ax.legend()
    ax.grid(True)

    # Keep equal aspect ratio
    max_range = np.array([X_value[:, 0].max() - X_value[:, 0].min(),
                          X_value[:, 1].max() - X_value[:, 1].min(),
                          X_value[:, 2].max() - X_value[:, 2].min()]).max() / 2.0

    mid_x = (X_value[:, 0].max() + X_value[:, 0].min()) * 0.5
    mid_y = (X_value[:, 1].max() + X_value[:, 1].min()) * 0.5
    mid_z = (X_value[:, 2].max() + X_value[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()


def plot_static_obstacles(X, rho, ax: Axes3D = None):
    N = X.shape[0]
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    colors = cm.Set1(np.linspace(0, 1, N))
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 5)
    for i in range(N):
        color = colors[i]
        pi = X[i]
        xs = pi[0] + rho * np.outer(np.cos(u), np.sin(v))
        ys = pi[1] + rho * np.outer(np.sin(u), np.sin(v))
        zs = pi[2] + rho * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(xs, ys, zs, color=color, alpha=0.15, linewidth=0, shade=True)
    return ax


def plot_dynamic_obstacles(X: np.ndarray, rho: float, ax: Axes3D = None):
    if len(X.shape) == 2:
        plot_static_obstacles(X, rho, ax)
        return
    T, N, _ = X.shape
    obs_pos = X[:, :, :3]
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d(-10, 10)
        ax.set_ylim3d(-10, 10)
        ax.set_zlim3d(-5, 5)
        ax.set_xlabel("X Position (km)", fontsize=13)
        ax.set_ylabel("Y Position (km)", fontsize=13)
        ax.set_zlabel("Z Position (km)", fontsize=13)
        ax.tick_params("both", labelsize=12)
        ax.set_box_aspect([1, 1, 0.5])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.view_init(elev=20, azim=45)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')

    color = "darkred"
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 5)
    for i in range(N):
        pi = X[:, i]
        ax.plot(pi[:, 0], pi[:, 1], pi[:, 2], linestyle="-", linewidth=2.5, color=color, alpha=0.8)

        p0 = obs_pos[0, i]
        xs = p0[0] + rho * np.outer(np.cos(u), np.sin(v))
        ys = p0[1] + rho * np.outer(np.sin(u), np.sin(v))
        zs = p0[2] + rho * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(xs, ys, zs, color=color, alpha=0.15, linewidth=0, shade=True)

        p1 = obs_pos[-1, i]
        xs = p1[0] + rho * np.outer(np.cos(u), np.sin(v))
        ys = p1[1] + rho * np.outer(np.sin(u), np.sin(v))
        zs = p1[2] + rho * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(xs, ys, zs, color=color, alpha=0.15, linewidth=0.1, shade=True)
        ax.scatter(p0[0], p0[1], p0[2], color=color, s=150, marker='o', edgecolors='black', linewidths=2, zorder=10)
        ax.scatter(p1[0], p1[1], p1[2], color=color, s=150, marker='s', edgecolors='black', linewidths=2, zorder=10)

    return ax


def plot_cluster_trajectories(X, rt, obs_pos=None, obs_rad=None, stat_config=False):
    N = len(X)
    fig: plt.Figure = plt.figure(figsize=(13, 10))
    ax: Axes3D = fig.add_subplot(111, projection='3d')

    """plot each satellite trajectory"""
    for i in range(N):
        xi = X[i]
        ax.plot(xi[:, 0], xi[:, 1], xi[:, 2], linewidth=2, alpha=0.7, c="b")
        ax.scatter(xi[0, 0], xi[0, 1], xi[0, 2], s=75, marker='o', edgecolors='black', linewidths=1, alpha=1, zorder=5)

    """plot obstacles"""
    if obs_pos is not None:
        plot_dynamic_obstacles(obs_pos, obs_rad, ax=ax)

    """plot target configuration"""
    if not stat_config:
        plot_tar_config(rt, ax)
    else:
        for i in range(N):
            ax.scatter(X[i][-1, 0], X[i][-1, 1], X[i][-1, 2], s=75, marker='*', edgecolors='black', linewidths=1,
                       alpha=1)

    """axis setting"""
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-5, 5])
    ax.set_xlabel('X Position (km)', fontsize=15, labelpad=10)
    ax.set_ylabel('Y Position (km)', fontsize=15, labelpad=10)
    ax.set_zlabel('Z Position (km)', fontsize=15, labelpad=10)
    ax.set_box_aspect([1, 1, 0.5])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.view_init(elev=35, azim=20)
    ax.tick_params(labelsize=13)

    """set legend"""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    custom_lines = [
        Line2D([0], [0], marker='o', color='gray', linestyle='None',
               markersize=10, markeredgecolor='black', markeredgewidth=2),
        Patch(facecolor='red', edgecolor='darkred', alpha=0.3)
    ]
    if stat_config:
        custom_lines.append(Line2D([0], [0], marker='*', color='red', linestyle='None',
                                   markersize=18, markeredgecolor='black', markeredgewidth=2))
        legend2 = ax.legend(custom_lines, ['Start Point', 'Obstacle', "target Point"], loc='upper right', fontsize=15,
                            framealpha=0.5)
    else:
        custom_lines.append(Patch(facecolor='blue', edgecolor='blue', alpha=0.2))
        legend2 = ax.legend(custom_lines, ['Start Point', 'Obstacle', "target Configuration"], loc='upper right',
                            fontsize=15, framealpha=0.5)
    ax.add_artist(legend2)
    plt.tight_layout()
    return fig, ax


def plot_trajectory_SCP(x_empc, x_empc_scp, Obs, rad, rt, x_scp=None):
    fig: plt.Figure = plt.figure(figsize=(13, 10))
    ax: Axes3D = fig.add_subplot(111, projection='3d')

    ax.scatter(x_empc[0, 0], x_empc[0, 1], x_empc[0, 2], s=75, marker='o', color="sienna",
               edgecolors='black', linewidths=1, alpha=1, label="Start Point")
    ax.plot(x_empc[:, 0], x_empc[:, 1], x_empc[:, 2], linewidth=2, c="b", label="Trajectory only by EMPC")
    ax.plot(x_empc_scp[:, 0], x_empc_scp[:, 1], x_empc_scp[:, 2], linewidth=2, c="orangered",
            label="Trajectory by EMPC-SCP")

    if x_scp is not None:
        ax.plot(x_scp[:, 0], x_scp[:, 1], x_scp[:, 2], linewidth=2, c="y", label="SCP revised Trajectory")

    """plot obstacles"""
    plot_dynamic_obstacles(Obs, rad, ax=ax)

    """plot target configuration"""
    plot_tar_config(rt, ax)

    """axis setting"""
    ax.tick_params(labelsize=12)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-5, 5])
    ax.set_xlabel('X Position (km)', fontsize=15, labelpad=10)
    ax.set_ylabel('Y Position (km)', fontsize=15, labelpad=10)
    ax.set_zlabel('Z Position (km)', fontsize=15, labelpad=10)
    ax.set_box_aspect([1, 1, 0.5])
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.view_init(elev=35, azim=20)

    ax.legend(loc="upper right", fontsize=15, framealpha=0.5)
    plt.tight_layout()
    pass


def plot_control_sequence(u_seq: np.ndarray):
    """
    u关于时间的序列变化图，绘制在一个axis中即可，ux随时间，uy随时间，和uz随时间，
    都表现在一个图里即可
    Args:
        u_seq: control sequence (N, 4): [ux, uy, uz, sign]
        sign表示当前的u是基于什么方法得到的，有0, 1两个取值，
        0代表这个u是EMPC得到的，1代表u是SCP得到的，需要在图中体现出来

    Returns:

    """
    from Global_Parameters import sat, op
    from matplotlib.patches import Patch

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'

    dt = sat.rm.get_sample_time()
    N = u_seq.shape[0]
    t_seq = np.arange(N) * dt
    sign = u_seq[:, 3].astype(int)

    scp_intervals = []
    start_idx = None
    for i in range(N):
        if sign[i] == 1 and start_idx is None:
            start_idx = i
        elif sign[i] != 1 and start_idx is not None:
            scp_intervals.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:
        scp_intervals.append((start_idx, N - 1))

    fig = plt.figure(figsize=(10, 5))
    ax: plt.Axes = fig.add_subplot(111)

    # SCP 触发时段背景着色
    scp_face = '#FFE2A8'
    for s, e in scp_intervals:
        ax.axvspan(t_seq[s] - dt / 2, t_seq[e] + dt / 2,
                   facecolor=scp_face, alpha=0.55, zorder=0)

    line_x, = ax.plot(t_seq, u_seq[:, 0], lw=2.0, color='#D62728', label=r'$u_x$')
    line_y, = ax.plot(t_seq, u_seq[:, 1], lw=2.0, color='#1F77B4', label=r'$u_y$')
    line_z, = ax.plot(t_seq, u_seq[:, 2], lw=2.0, color='#2CA02C', label=r'$u_z$')
    ax.set_xlim(float(t_seq[0]), float(t_seq[-1] + dt))
    ax.set_ylim(-op.u_max * 1.1, op.u_max * 1.1)

    ax.set_xlabel('Time (s)', fontsize=17)
    ax.set_ylabel(r'Control Acceleration $u$ (km/s$^2$)', fontsize=17)
    ax.tick_params(labelsize=15)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
    handles = [line_x, line_y, line_z,
               Patch(facecolor=scp_face, alpha=0.55, edgecolor='none', label='SCP active')]
    ax.legend(handles=handles, loc='upper right', fontsize=15, framealpha=0.9)

    plt.tight_layout()
    plt.show()


def plot_tar_config(rt, ax):
    r_tube = rt / 10.0
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, 2 * np.pi, 50)
    u, v = np.meshgrid(u, v)
    x_spine = rt / 2 * np.cos(u)
    y_spine = rt * np.sin(u)
    nx = rt * np.cos(u)
    ny = rt / 2 * np.sin(u)
    norm_n = np.sqrt(nx ** 2 + ny ** 2)
    nx_unit = nx / norm_n
    ny_unit = ny / norm_n
    X = x_spine + r_tube * np.cos(v) * nx_unit
    Y = y_spine + r_tube * np.cos(v) * ny_unit
    Z = r_tube * np.sin(v)
    ax.plot_surface(X, Y, Z, color='blue', alpha=0.2, label="target configuration")


def plot_distance_with_obstacle(d1, d2, ro=0.6):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16

    N = len(d1)
    obstacles = np.arange(1, N + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(obstacles, d1, color='#D62728', linestyle='--', marker='o', markersize=10,
            linewidth=2.5, label='Trajectory by EMPC')
    ax.plot(obstacles, d2, color='#1F77B4', linestyle='-', marker='s', markersize=10,
            linewidth=2.5, label='Trajectory by EMPC-SCP')
    ax.axhline(y=ro, color='#2CA02C', linestyle='-.', linewidth=2.5, label=f'Safety Distance ($r_o$={ro})')
    ax.fill_between(obstacles, 0, ro, color='#D62728', alpha=0.1)
    ax.set_xlabel('Obstacle No.')
    ax.set_ylabel('Minimum Distance(km)')
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    y_min = min(0, min(d1), min(d2))
    y_max = max(max(d1), max(d2))
    ax.set_ylim(bottom=y_min, top=y_max * 1.1)
    ax.set_xlim(0.95, N + 0.05)
    ax.grid(True, linestyle=':', color='gray', alpha=0.6)
    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    plt.show()


def plot_distance_between_sat(d1, d2, rs=0.2):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16

    N = len(d1)
    sats = np.arange(1, N + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sats, d1, color='#D62728', linestyle='--', marker='o', markersize=10, linewidth=2.5,
            label='Trajectory by EMPC')
    ax.plot(sats, d2, color='#1F77B4', linestyle='-', marker='s', markersize=10, linewidth=2.5,
            label='Trajectory by EMPC-SCP')
    ax.axhline(y=rs, color='#2CA02C', linestyle='-.', linewidth=2.5, label=f'Safety Distance ($r_s=${rs})')
    ax.fill_between(sats, 0, rs, color='#D62728', alpha=0.1)
    ax.set_xlabel('Satellite No.')
    ax.set_ylabel('Minimum Distance(km)')
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    y_max = max(max(d1), max(d2))
    ax.set_ylim(bottom=0, top=y_max * 1.1)
    ax.set_xlim(0.95, N + 0.05)
    ax.grid(True, linestyle=':', color='gray', alpha=0.6)
    ax.legend(loc='center right', frameon=False)
    plt.tight_layout()
    plt.show()


def plot_computation_time_comparison(save_path="algorithm_comparison.pdf"):
    N = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    t_nmpc = [2.0, 11.1, 59.3, 177.4, 402, 1058, 1814, 2896, 4374, 6952,
              np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    t_dmpc = [1.92, 2.03, 2.08, 2.15, 2.2, 2.26, 2.32, 2.38, 2.45, 2.54,
              3.2, 4.1, 5.3, 6.8, 8.3, 9.8, 11.2, 13.9, 16.7]
    t_empc = [0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.06, 0.07, 0.08, 0.10,
              0.14, 0.2, 0.28, 0.37, 0.48, 0.60, 0.71, 0.82, 0.95]

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.rcParams['legend.fontsize'] = 15

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(N, t_nmpc,
            color='#D62728', marker='o', linestyle='-', linewidth=2, markersize=7, label='Nonlinear MPC')
    ax.plot(N, t_dmpc,
            color='#1F77B4', marker='s', linestyle='-', linewidth=2, markersize=7, label='Decentralized MPC')
    ax.plot(N, t_empc,
            color='#2CA02C', marker='^', linestyle='-', linewidth=2, markersize=7, label='Explicit MPC with SCP')
    ax.set_xscale('log')

    ax1: plt.Axes = ax.inset_axes((0.57, 0.2, 0.4, 0.5))
    ax1.plot(N, t_dmpc,
             color='#1F77B4', marker='s', linestyle='-', linewidth=2, markersize=7,
             label='Decentralized MPC')
    ax1.plot(N, t_empc,
             color='#2CA02C', marker='^', linestyle='-', linewidth=2, markersize=7,
             label='Explicit MPC with SCP')
    ax1.set_xscale('log')
    ax1.grid(True, which='major', linestyle='-', color='gray', alpha=0.3)
    ax1.grid(True, which='minor', linestyle=':', color='gray', alpha=0.2)

    ax.set_xlabel('Satellite Numbers ($N$)')
    ax.set_ylabel('Computation Time (s)')
    ax.set_xlim(0.9, 105)  # X 轴稍微留白
    ax.grid(True, which='major', linestyle='-', color='gray', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', color='gray', alpha=0.2)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='black')

    plt.tight_layout()
    plt.savefig(save_path, format=save_path.split('.')[-1], bbox_inches='tight')
    plt.show()
    pass


def plot_diff_algo_tra(x_empc, x_dmpc, x_mpc, Obs, rad, rt):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    c_empc = '#1F77B4'  # 深蓝色 (Explicit MPC)
    c_dmpc = '#D62728'  # 砖红色 (Decentralized MPC)
    c_mpc = '#2CA02C'  # 森林绿 (Nonlinear MPC)
    c_start = '#FF7F0E'  # 活力橙 (起点)
    c_target = '#9467BD'  # 深紫色 (终点)

    ax.scatter(x_empc[0, 0], x_empc[0, 1], x_empc[0, 2],
               s=120, marker='o', color=c_start,
               edgecolors='black', linewidths=1.5, alpha=1, zorder=5, label="Start Point")

    ax.plot(x_empc[:, 0], x_empc[:, 1], x_empc[:, 2],
            lw=2.5, c=c_empc, linestyle='-', zorder=4, label="Explicit MPC with SCP")
    ax.plot(x_dmpc[:, 0], x_dmpc[:, 1], x_dmpc[:, 2],
            lw=2.5, c=c_dmpc, linestyle='--', zorder=3, label="Decentralized MPC")
    ax.plot(x_mpc[:, 0], x_mpc[:, 1], x_mpc[:, 2],
            lw=2.5, c=c_mpc, linestyle='-.', zorder=2, label="Nonlinear MPC")

    """plot obstacles"""
    plot_dynamic_obstacles(Obs, rad, ax=ax)

    """plot target configuration"""
    if np.isscalar(rt):
        plot_tar_config(rt, ax)
    else:
        ax.scatter(rt[0], rt[1], rt[2], s=240, marker='*', color=c_target,
                   edgecolors='red', linewidths=1.5, alpha=1, zorder=5, label="Target Point")

    ax.tick_params(labelsize=12, pad=5)  # 减小一点字号，让画面更留白
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-5, 5])

    ax.set_xlabel('X Position (km)', fontsize=14, labelpad=12)
    ax.set_ylabel('Y Position (km)', fontsize=14, labelpad=12)
    ax.set_zlabel('Z Position (km)', fontsize=14, labelpad=12)

    ax.set_box_aspect([1, 1, 0.5])
    ax.grid(True, alpha=0.4, linestyle=':', color='gray')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    ax.view_init(elev=35, azim=20)
    ax.legend(loc="center left", bbox_to_anchor=(0.6, 0.7), fontsize=14, framealpha=0.9, edgecolor='lightgray')
    plt.tight_layout(rect=[0, 0, 1, 1])

    # 建议保存为 PDF 格式
    # plt.savefig('trajectory_3d.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def plot_consumption():
    Method = ["Explicit MPC", "Decentralized MPC", "Nonlinear MPC"]
    U_st = [0.011015353661484156, 0.010493229175166870, 0.008884428038197627]
    U_dy = [0.009395268739288552, 0.009370855910490179, 0.009034138495369104]
    y_label = r"Consumption $\sum \|u_k\|$"

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.fontsize'] = 17

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(Method))  # X 轴的基准位置
    width = 0.35  # 柱子的宽度

    color_st = '#4682B4'  # Steel Blue
    color_dy = '#CD5C5C'  # Indian Red

    rects1 = ax.bar(x - width / 2, U_st, width, label='Static Target Scenario',
                    color=color_st, edgecolor='black', linewidth=1.2, zorder=3)
    rects2 = ax.bar(x + width / 2, U_dy, width, label='Dynamic Target Scenario',
                    color=color_dy, edgecolor='black', linewidth=1.2, zorder=3)

    # ------------------ 坐标轴与细节美化 ------------------
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(Method)

    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    max_y = max(max(U_st), max(U_dy))
    ax.set_ylim(0, max_y * 1.15)
    ax.legend(loc='upper right', frameon=False)

    def autolabel(rects):
        """在每个柱子上方居中显示具体的数值"""
        for rect in rects:
            height = rect.get_height()
            # 保留4位小数，足以展示动态场景下的能耗差距
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 垂直偏移 4 个像素
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=15, fontfamily='Times New Roman')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    # plt.savefig('energy_consumption_comparison.pdf', format='pdf', bbox_inches='tight')
    plt.show()
