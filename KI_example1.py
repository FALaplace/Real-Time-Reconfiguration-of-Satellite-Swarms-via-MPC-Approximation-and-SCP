import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, Matern
from Kernel_Interpolation_Class import Kernel_Interpolation_GPR_based


# 使用示例
def kernel_interpolation_example():
    # 生成示例数据
    np.random.seed(42)
    X_train = np.linspace(0, 10, 21).reshape(-1, 1)
    y_train = np.sin(X_train).ravel() + 0.01 * np.random.randn(21)

    # 创建测试数据
    X_test = np.linspace(0, 10, 101).reshape(-1, 1)
    y_true = np.sin(X_test).ravel()

    # 使用不同的核函数进行插值
    kernels = {
        'Length Scale=0.1': Matern(length_scale=0.1, nu=1.5),
        'Length Scale=1': Matern(length_scale=1, nu=1.5),
        'Length Scale=10': Matern(length_scale=10, nu=1.5)
    }

    plt.figure(figsize=(12, 8))

    for i, (name, kernel) in enumerate(kernels.items(), 1):
        # 创建核插值模型
        model = Kernel_Interpolation_GPR_based(kernel=kernel)

        # 拟合模型并获取系数
        coeffs = model.fit(X_train, y_train)

        # 计算训练点的协方差矩阵
        K_train, L_train = model.covariance_matrix(X_train)

        # 计算测试点与训练点之间的核矩阵
        K_test_train = kernel(X_test, X_train)

        # 进行预测
        y_pred = K_test_train @ coeffs

        # 计算预测的不确定性（可选）
        v = cho_solve((L_train, GPR_CHOLESKY_LOWER), K_test_train.T)
        y_cov = kernel(X_test) - K_test_train @ v

        # 绘制结果
        plt.subplot(2, 2, i)
        plt.scatter(X_train, y_train, c='red', label='Training points', zorder=5)
        plt.plot(X_test, y_true, 'k--', label='True function', alpha=0.7)
        plt.plot(X_test, y_pred, label=f'Prediction ({name})')
        plt.fill_between(
            X_test.ravel(),
            y_pred - 2 * np.sqrt(np.diag(y_cov)),
            y_pred + 2 * np.sqrt(np.diag(y_cov)),
            alpha=0.2, label='Uncertainty'
        )
        plt.title(f'Kernel Interpolation with {name}')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')

    # 显示系数信息
    plt.subplot(2, 2, 4)
    model = Kernel_Interpolation_GPR_based(kernel=RBF(length_scale=1.0))
    coeffs = model.fit(X_train, y_train)
    plt.stem(coeffs)
    plt.title('RKHS Coefficients')
    plt.xlabel('Sample index')
    plt.ylabel('Coefficient value')

    plt.tight_layout()
    plt.show()

    return coeffs


# 运行示例
if __name__ == "__main__":
    # 需要导入的额外依赖
    from scipy.linalg import cho_solve

    # 定义常量（根据您的实际代码）
    GPR_CHOLESKY_LOWER = True

    # 运行示例
    coefficients = kernel_interpolation_example()
    print(f"拟合得到的系数数量: {len(coefficients)}")
    print(f"系数范围: [{coefficients.min():.3f}, {coefficients.max():.3f}]")