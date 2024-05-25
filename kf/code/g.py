import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保每次生成的数据都相同
np.random.seed(43)

# 真实轨迹模拟
kx = 0.01
ky = 0.05
g = 9.8
t = 15
Ts = 0.1
len = int(t/Ts)
dax = 3
day = 3
X = np.zeros((len, 4))
X[0, :] = [0, 50, 500, 0]
for k in range(1, len):
    x, vx, y, vy = X[k-1, :]
    x += vx * Ts
    vx += (-kx * vx**2 + dax * np.random.randn()) * Ts
    y += vy * Ts
    vy += (ky * vy**2 - g + day * np.random.randn()) * Ts
    X[k, :] = [x, vx, y, vy]

# 构造量测量
dr = 8
dafa = 0.1
Z = np.zeros((len, 2))
for k in range(len):
    r = np.sqrt(X[k, 0]**2 + X[k, 2]**2) + dr * np.random.randn()
    a = np.arctan(X[k, 0] / X[k, 2]) * 57.3 + dafa * np.random.randn()
    Z[k, :] = [r, a]

# ekf 滤波
Qk = np.diag([0, dax/10, 0, day/10])**2
Rk = np.diag([dr, dafa])**2
Pk = 10 * np.eye(4)
Pkk_1 = 10 * np.eye(4)
x_hat = np.array([0, 40, 400, 0])
X_est = np.zeros((len, 4))
x_forecast = np.zeros(4)
z = np.zeros(4)
for k in range(len):
    # 状态预测
    x1 = x_hat[0] + x_hat[1] * Ts
    vx1 = x_hat[1] + (-kx * x_hat[1]**2) * Ts
    y1 = x_hat[2] + x_hat[3] * Ts
    vy1 = x_hat[3] + (ky * x_hat[3]**2 - g) * Ts
    x_forecast = np.array([x1, vx1, y1, vy1])
    # 观测预测
    r = np.sqrt(x1**2 + y1**2)
    alpha = np.arctan(x1 / y1) * 57.3
    y_yuce = np.array([r, alpha])
    # 状态矩阵
    vx = x_forecast[1]
    vy = x_forecast[3]
    F = np.zeros((4, 4))
    F[0, 0] = 1
    F[0, 1] = Ts
    F[1, 1] = 1 - 2 * kx * vx * Ts
    F[2, 2] = 1
    F[2, 3] = Ts
    F[3, 3] = 1 + 2 * ky * vy * Ts
    Pkk_1 = F @ Pk @ F.T + Qk
    # 观测矩阵
    x = x_forecast[0]
    y = x_forecast[2]
    H = np.zeros((2, 4))
    r = np.sqrt(x**2 + y**2)
    xy2 = 1 + (x / y)**2
    H[0, 0] = x / r
    H[0, 2] = y / r
    H[1, 0] = (1 / y) / xy2
    H[1, 2] = (-x / y**2) / xy2
    Kk = Pkk_1 @ H.T @ np.linalg.inv(H @ Pkk_1 @ H.T + Rk)
    x_hat = x_forecast + Kk @ (Z[k, :] - y_yuce)
    Pk = (np.eye(4) - Kk @ H) @ Pkk_1
    X_est[k, :] = x_hat

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(X[:, 0], X[:, 2], '-b', label='Real')
plt.plot(Z[:, 0] * np.sin(Z[:, 1] * np.pi / 180), Z[:, 0] * np.cos(Z[:, 1] * np.pi / 180), label='Measurement')
plt.plot(X_est[:, 0], X_est[:, 2], 'r', label='EKF Estimated')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('EKF Simulation')
plt.legend()
plt.axis([-5, 230, 290, 530])
plt.grid(True)
plt.show()
