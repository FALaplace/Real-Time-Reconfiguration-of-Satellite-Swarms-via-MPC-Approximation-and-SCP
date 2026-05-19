import numpy as np
from Global_Parameters import op


def AKI_grid(D, Mp, Mv):
    """
    D: ndarray of shape (n, 2)
    Mp: number of grid points per position dimension
    Mv: number of grid points per velocity dimension
    return: ndarray of shape (Mp^(n/2) * Mv^(n/2), n)
    """
    n = D.shape[0]
    half = n // 2
    p_grid = [np.linspace(D[i, 0], D[i, 1], Mp) for i in range(half)]
    v_grid = [np.linspace(D[i + half, 0], D[i + half, 1], Mv) for i in range(half)]
    grids = p_grid + v_grid
    mesh = np.meshgrid(*grids, indexing='ij')
    X = np.stack(mesh, axis=-1).reshape(-1, n)
    return X


def inv_grid(a, flag="pos") -> int:
    if flag == "pos":
        bounds = np.linspace(-op.pos_max, op.pos_max, 5)
    elif flag == "vel":
        bounds = np.linspace(-op.vel_max, op.vel_max, 5)
    else:
        return -1
    match a:
        case _ if a < bounds[1]:
            return 0
        case _ if bounds[1] <= a < bounds[2]:
            return 1
        case _ if bounds[2] <= a < bounds[3]:
            return 2
        case _:
            return 3


def find_local_samples(x):
    idx = []
    for i in range(x.size):
        if i <= 2:
            idx.append(inv_grid(x[i], "pos"))
        else:
            idx.append(inv_grid(x[i], "vel"))
    idx = tuple(idx)
    return idx
