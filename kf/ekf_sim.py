import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 设置随机种子以确保每次生成的数据都相同
# np.random.seed(43)

# 真实轨迹模拟
t = 50
Ts = 0.01
len_t = int(t / Ts)
X = np.zeros((len_t, 4))
X[0, :] = [5, 0.5, 9, 0.5]

dx = 0.001
dvx = 0.001
dy = 0.001
dvy = 0.001

# 固定圆心坐标
center_x = 5
center_y = 5
radius = 4

for k in range(1, len_t):
    vx = X[k - 1, 1]
    vy = X[k - 1, 3]

    # 计算当前角度
    theta = np.arctan2(X[k - 1, 2] - center_y, X[k - 1, 0] - center_x)

    # 计算新位置
    theta_new = theta + vx * Ts
    x = center_x + radius * np.cos(theta_new) + np.random.normal(0, dx)
    y = center_y + radius * np.sin(theta_new) + np.random.normal(0, dy)

    # 计算新速度
    vx_new = vx + np.random.normal(0, dvx)
    vy_new = vy + np.random.normal(0, dvy)

    X[k, :] = [x, vx_new, y, vy_new]

# 构造量测量
dr = 0.15
da = 1.5
Z = np.zeros((len_t, 2))
for k in range(len_t):
    r = np.sqrt(X[k, 0] ** 2 + X[k, 2] ** 2) + np.random.normal(0, dr)
    a = np.arctan(X[k, 0] / X[k, 2]) * 57.3 + np.random.normal(0, da)
    Z[k, :] = [r, a]

# ekf 滤波
# Qk = np.diag([dx, dvx, dy, dvy]) ** 2
Qk = np.diag([0.0001, 0.0001, 0.0001, 0.0001])
# Rk = np.diag([dr, da]) ** 2
Rk = np.diag([0.1, 1])
Pk = np.eye(4)
P_forecast = np.eye(4)
x_hat = np.array([5, 0.5, 9, 0.5])
X_est = np.zeros((len_t, 4))
x_forecast = np.zeros(4)
z = np.zeros(4)
F = np.zeros((4, 4))
F[0, 0] = 1
F[0, 1] = Ts
F[1, 1] = 1
F[2, 2] = 1
F[2, 3] = Ts
F[3, 3] = 1
for k in range(len_t):
    # 先验
    x1 = x_hat[0] + x_hat[1] * Ts
    vx1 = x_hat[1]
    y1 = x_hat[2] + x_hat[3] * Ts
    vy1 = x_hat[3]
    x_forecast = np.array([x1, vx1, y1, vy1])
    # 观测预测
    r = np.sqrt(x1 ** 2 + y1 ** 2)
    alpha = np.arctan(x1 / y1) * 57.3
    z = np.array([r, alpha])
    # 先验估计协方差矩阵
    P_forecast = F @ Pk @ F.T + Qk
    # 观测矩阵H
    x = x_forecast[0]
    y = x_forecast[2]
    H = np.zeros((2, 4))
    r = np.sqrt(x ** 2 + y ** 2)
    xy2 = 1 + (x / y) ** 2
    H[0, 0] = x / r
    H[0, 2] = y / r
    H[1, 0] = (1 / y) / xy2
    H[1, 2] = (-x / y ** 2) / xy2
    # 卡尔曼增益
    Kk = P_forecast @ H.T @ np.linalg.inv(H @ P_forecast @ H.T + Rk)
    # 后验
    x_hat = x_forecast + Kk @ (Z[k, :] - z)
    # 更新
    Pk = (np.eye(4) - Kk @ H) @ P_forecast
    X_est[k, :] = x_hat

# 创建画布和轴
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('EKF Simulation')

# 初始化绘图对象
real_line, = ax.plot([], [], '-b', label='Real')
measurement_line, = ax.plot([], [], 'x', color='orange', label='Measurement')
ekf_line, = ax.plot([], [], 'r', label='EKF Estimated')
ax.legend()
ax.grid(True)


# 更新函数
def update(frame):
    start_frame = max(0, frame - 500)  # 计算起始帧
    end_frame = min(len(X), frame)  # 计算结束帧
    real_line.set_data(X[start_frame:end_frame, 0], X[start_frame:end_frame, 2])
    measurement_line.set_data(Z[start_frame:end_frame, 0] * np.sin(Z[start_frame:end_frame, 1] * np.pi / 180),
                              Z[start_frame:end_frame, 0] * np.cos(Z[start_frame:end_frame, 1] * np.pi / 180))
    ekf_line.set_data(X_est[start_frame:end_frame, 0], X_est[start_frame:end_frame, 2])
    return real_line, measurement_line, ekf_line

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=len_t, blit=True, interval=Ts)

plt.show()
