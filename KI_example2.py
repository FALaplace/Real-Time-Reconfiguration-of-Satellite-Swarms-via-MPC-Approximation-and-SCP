import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, Matern
from scipy.interpolate import lagrange, CubicSpline
from scipy.linalg import cholesky, cho_solve
from Kernel_Interpolation_Class import Kernel_Interpolation_GPR_based

# 假设这些已经在您的代码中定义
GPR_CHOLESKY_LOWER = True


def runge_function(x):
    return 1 / (1 + 25 * x ** 2)


def compare_interpolation_methods():
    """
    对比核插值和多项式插值在sin(x)函数上的表现
    """
    np.random.seed(42)

    # 生成训练数据
    n_train = 8  # 少量训练点，便于观察插值行为
    X_train = np.linspace(0, 2 * np.pi, n_train).reshape(-1, 1)
    y_train = runge_function(X_train).ravel()

    # 生成密集的测试点
    X_test = np.linspace(-0.5, 2 * np.pi + 0.5, 200).reshape(-1, 1)
    y_true = runge_function(X_test).ravel()

    # 定义不同的插值方法
    methods = {}

    # 1. 拉格朗日多项式插值
    try:
        poly_lagrange = lagrange(X_train.ravel(), y_train)
        y_poly_lagrange = poly_lagrange(X_test.ravel())
        methods['Lagrange Interpolation'] = y_poly_lagrange
    except Exception as e:
        print(f"拉格朗日插值失败: {e}")

    # 2. 三次样条插值
    try:
        spline = CubicSpline(X_train.ravel(), y_train)
        y_spline = spline(X_test.ravel())
        methods['CubicSpline'] = y_spline
    except Exception as e:
        print(f"三次样条插值失败: {e}")

    # 3. 核插值 - RBF核
    kernel_rbf = Matern(length_scale=1.0, nu=0.5)
    model_rbf = Kernel_Interpolation_GPR_based(kernel_rbf)
    coff_rbf = model_rbf.fit(X_train, y_train)
    y_rbf = model_rbf.kernel(X_test, X_train) @ coff_rbf
    methods['Matern05-1'] = y_rbf

    # 4. 核插值 - Matern核 (nu=1.5)
    kernel_matern = Matern(length_scale=0.2, nu=1.5)
    model_matern = Kernel_Interpolation_GPR_based(kernel_matern)
    coff_m321 = model_matern.fit(X_train, y_train)
    y_matern = model_matern.kernel(X_test, X_train) @ coff_m321
    methods['Matern32-02'] = y_matern

    # 5. 核插值 - Matern核 (nu=0.5)
    kernel_matern05 = Matern(length_scale=1, nu=1.5)
    model_matern05 = Kernel_Interpolation_GPR_based(kernel_matern05)
    coff_ma05 = model_matern05.fit(X_train, y_train)
    y_matern05 = model_matern05.kernel(X_test, X_train) @ coff_ma05
    methods['Matern32-1'] = y_matern05

    # 绘制对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))

    for idx, (method_name, y_pred) in enumerate(methods.items()):
        ax = axes[idx]

        # 绘制真实函数
        ax.plot(X_test, y_true, 'k-', linewidth=2, alpha=0.7, label='true value')

        # 绘制插值结果
        ax.plot(X_test, y_pred, linewidth=2, color=colors[idx], label=f'{method_name}')

        # 绘制训练点
        ax.scatter(X_train, y_train, c='red', s=80, zorder=5, label='sample point')

        # 计算误差
        error = np.mean((y_pred - y_true) ** 2)
        ax.set_title(f'{method_name}\nMSE: {error:.6f}', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 2 * np.pi + 0.5)
        ax.set_ylim(-2, 2)

    # 最后一个子图显示所有方法对比
    ax = axes[-1]
    ax.plot(X_test, y_true, 'k-', linewidth=3, alpha=0.7, label='original function')
    for idx, (method_name, y_pred) in enumerate(methods.items()):
        ax.plot(X_test, y_pred, linewidth=1.5, color=colors[idx], label=method_name, alpha=0.8)
    ax.scatter(X_train, y_train, c='red', s=80, zorder=5, label='sample point')
    ax.set_title('comparison of all methods')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 2 * np.pi + 0.5)
    ax.set_ylim(-2, 2)

    plt.tight_layout()
    plt.show()

    # 打印误差统计
    print("\n各方法均方误差(MSE)对比:")
    print("-" * 40)
    for method_name, y_pred in methods.items():
        mse = np.mean((y_pred - y_true) ** 2)
        max_error = np.max(np.abs(y_pred - y_true))
        print(f"{method_name:20} MSE: {mse:.6f}, 最大误差: {max_error:.4f}")

    return methods, X_train, y_train, X_test, y_true


compare_interpolation_methods()
