import numpy as np
import matplotlib.pyplot as plt

# 参数设置
x_min, x_max = -1, 6
nx = 200
dx = (x_max - x_min) / (nx - 1)
x = np.linspace(x_min, x_max, nx)
dt = 0.005
nt = int(6 / dt)

# 初始条件
def initial_condition(x):
    u = np.zeros_like(x)
    u[(x > 0) & (x < 1)] = 1
    # u[(x >= 1) & (x < 2)] = 1
    return u

# Lax-Friedrichs
def lax_friedrichs(u, dt, dx):
    u_next = np.zeros_like(u)
    f = 0.5 * u ** 2
    u_next[1:-1] = 0.5 * (u[:-2] + u[2:]) - dt / (2 * dx) * (f[2:] - f[:-2])
    return u_next

# Richtmyer
def richtmyer(u, dt, dx):
    u_half = np.zeros_like(u)
    f = 0.5 * u ** 2
    u_half[1:-1] = 0.5 * (u[:-2] + u[2:]) - dt / (2 * dx) * (f[2:] - f[:-2])
    f_half = 0.5 * u_half ** 2
    u_next = np.zeros_like(u)
    u_next[1:-1] = u[1:-1] - dt / dx * (f_half[2:] - f_half[:-2])
    return u_next

# MacCormack
def mac_cormack(u, dt, dx):
    u_star = np.copy(u)
    f = 0.5 * u ** 2
    u_star[:-1] = u[:-1] - dt / dx * (f[1:] - f[:-1])
    f_star = 0.5 * u_star ** 2
    u_next = np.copy(u)
    u_next[1:] = 0.5 * (u[1:] + u_star[1:] - dt / dx * (f_star[1:] - f_star[:-1]))
    return u_next

# Godunov
def godunov(u, dx, dt):
    u_next = np.zeros_like(u)
    f = lambda u: 0.5 * u ** 2
    for i in range(1, len(u) - 1):
        uL = u[i - 1]
        uR = u[i]
        s = (uL + uR) / 2  # 中心速度
        if uL <= uR:
            if s >= 0:
                flux_left = f(uL)
            else:
                flux_left = f(uR)
        else:
            flux_left = max(f(uL), f(uR)) if uL > uR else 0
        
        uL = u[i]
        uR = u[i + 1]
        s = (uL + uR) / 2
        if uL <= uR:
            if s >= 0:
                flux_right = f(uL)
            else:
                flux_right = f(uR)
        else:
            flux_right = max(f(uL), f(uR)) if uL > uR else 0
        
        u_next[i] = u[i] - dt / dx * (flux_right - flux_left)
    return u_next


# 精确解（直接用稀疏波 + 激波拼接）
def exact_solution(x, t):
    u = np.zeros_like(x)
    if t <= 2:
        u[(x >= 0) & (x <= t)] = x[(x >= 0) & (x <= t)] / t
        u[(x > t) & (x <= 1 + t/2)] = 1.0
        u[x > 1 + t/2] = 0.0
    else:
        u[(x >= 0) & (x <= np.sqrt(2*t))] = x[(x >= 0) & (x <= np.sqrt(2*t))] / t
        u[x > np.sqrt(2*t)] = 0.0
    return u

# 主循环
def solve_and_plot():
    times = [0, 1, 2, 3, 4, 5, 6]
    u_lf = initial_condition(x)
    u_ri = initial_condition(x)
    u_mc = initial_condition(x)
    u_gd = initial_condition(x)

    for n in range(1, nt + 1):
        u_lf = lax_friedrichs(u_lf, dt, dx)
        u_ri = richtmyer(u_ri, dt, dx)
        u_mc = mac_cormack(u_mc, dt, dx)
        u_gd = godunov(u_gd, dx, dt)

        t = n * dt
        if np.any(np.isclose(t, times, atol=0.001)):
            u_exact = exact_solution(x, t)

            plt.figure(figsize=(8, 5))
            plt.plot(x, u_exact, 'k-', label='Exact')
            plt.plot(x, u_lf, 'b--', label='Lax-Friedrichs')
            plt.plot(x, u_ri, 'r--', label='Richtmyer')
            plt.plot(x, u_mc, 'g--', label='MacCormack')
            plt.plot(x, u_gd, 'm--', label='Godunov')
            plt.title(f'Time = {t:.1f}')
            plt.xlabel('x')
            plt.ylabel('u')
            plt.ylim(-0.2, 2.5)
            plt.legend()
            plt.grid()
            plt.show()

solve_and_plot()
