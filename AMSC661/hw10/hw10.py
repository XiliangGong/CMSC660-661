import numpy as np
import matplotlib.pyplot as plt

# 参数设定
a = np.sqrt(2)
h = 0.05  # 空间步长
x = np.arange(-6, 6 + h, h)
N = len(x)
T_values = [1/(2*a), 1/a, 2/a, 4/a]

# 初始条件
phi = np.maximum(1 - np.abs(x), 0)
psi = np.zeros_like(x)

# phi'(x)
phi_prime = np.zeros_like(x)
phi_prime[(x > -1) & (x < 0)] = 1
phi_prime[(x > 0) & (x < 1)] = -1

# 初始条件下的 ξ 和 η
xi0 = 0.5 * phi_prime
eta0 = 0.5 * phi_prime

# 周期边界条件辅助函数
def periodic(u):
    u[0] = u[-2]
    u[-1] = u[1]
    return u

# 数值格式函数
def lax_friedrichs(u, c, k, h):
    u_new = np.zeros_like(u)
    u = periodic(u)
    u_new[1:-1] = 0.5 * (u[2:] + u[:-2]) - (k * c / (2 * h)) * (u[2:] - u[:-2])
    return u_new

def upwind(u, c, k, h):
    u_new = np.zeros_like(u)
    u = periodic(u)
    if c > 0:
        u_new[1:-1] = u[1:-1] - (k * c / h) * (u[1:-1] - u[:-2])
    else:
        u_new[1:-1] = u[1:-1] - (k * c / h) * (u[2:] - u[1:-1])
    return u_new

def lax_wendroff(u, c, k, h):
    u_new = np.zeros_like(u)
    u = periodic(u)
    alpha = c * k / h
    u_new[1:-1] = u[1:-1] - 0.5 * alpha * (u[2:] - u[:-2]) + 0.5 * alpha**2 * (u[2:] - 2 * u[1:-1] + u[:-2])
    return u_new

# 精确解 d'Alembert
def exact_solution(x, t, phi, psi, a):
    x_plus = np.mod(x + a * t + 6, 12) - 6
    x_minus = np.mod(x - a * t + 6, 12) - 6
    phi_interp = np.interp(x_plus, x, phi) + np.interp(x_minus, x, phi)
    psi_integral = np.zeros_like(x)
    for i in range(len(x)):
        xi_range = np.linspace(x_minus[i], x_plus[i], 100)
        psi_integral[i] = np.trapz(np.interp(xi_range, x, psi), xi_range)
    return 0.5 * phi_interp + psi_integral / (2 * a)

# 主数值模拟函数
def simulate(method_xi, method_eta, k, T_max):
    xi = xi0.copy()
    eta = eta0.copy()
    t = 0
    while t < T_max:
        xi = method_xi(xi, a, k, h)
        eta = method_eta(eta, -a, k, h)
        t += k
    # 恢复 w 和 u
    w1 = a * xi - a * eta
    w2 = xi + eta
    u_num = np.zeros_like(w2)
    u_num[1:] = np.cumsum(w2[1:]) * h
    u_num = u_num - np.mean(u_num) + np.mean(phi)  # 平衡常数
    return u_num

# 模拟和绘图
k = 0.01  # 时间步长
methods = {
    "Lax-Friedrichs": (lax_friedrichs, lax_friedrichs),
    "Upwind": (upwind, upwind),
    "Lax-Wendroff": (lax_wendroff, lax_wendroff)
}

for T in T_values:
    plt.figure(figsize=(12, 6))
    exact_u = exact_solution(x, T, phi, psi, a)
    plt.plot(x, exact_u, 'k-', label='Exact', linewidth=2)
    for name, (mxi, meta) in methods.items():
        u_num = simulate(mxi, meta, k, T)
        plt.plot(x, u_num, label=f'{name}')
    plt.title(f'Time t = {T:.3f}')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.legend()
    plt.grid(True)
    plt.show()
