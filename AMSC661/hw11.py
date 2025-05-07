import numpy as np
import matplotlib.pyplot as plt

# 参数设置
x_min, x_max = -1, 6
nx = 200
dx = (x_max - x_min) / (nx - 1)
x = np.linspace(x_min, x_max, nx)
dt = 0.005
nt = int(5 / dt)

# 初始条件
def initial_condition(x):
    u = np.zeros_like(x)
    u[(x > 0) & (x < 1)] = 2
    u[(x >= 1) & (x < 2)] = 1
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

# 精确解（直接用稀疏波 + 激波拼接）
def exact_solution(x, t):
    u = np.zeros_like(x)
    if t < 1:
        for i, xi in enumerate(x):
            if xi < 0:
                u[i] = 0
            elif xi < 2*t:
                u[i] = xi / t
            elif xi < 1 + 1.5 * t:
                u[i] = 2
            elif xi < 1.5 +  t:
                u[i] =1
            else:
                u[i] =0
    elif t <1.5:
        for i, xi in enumerate(x):
            if xi <0:
                u[i]=0
            elif xi<2*t:
                u[i]=xi/t
            elif xi<2 + 0.5*t:
                u[i]=2
            else:
                u[i]=0
    else:
        for i, xi in enumerate(x):
            if xi <0:
                u[i]=0
            elif xi< np.sqrt(6*t):
                u[i]=xi/t
            else:
                u[i]=0
    return u

# 主循环
def solve_and_plot():
    times = [0.5, 1.5, 2.5, 3.5, 5.0]
    u_lf = initial_condition(x)
    u_ri = initial_condition(x)
    u_mc = initial_condition(x)

    for n in range(1, nt + 1):
        u_lf = lax_friedrichs(u_lf, dt, dx)
        u_ri = richtmyer(u_ri, dt, dx)
        u_mc = mac_cormack(u_mc, dt, dx)

        t = n * dt
        if np.any(np.isclose(t, times, atol=0.001)):
            u_exact = exact_solution(x, t)

            plt.figure(figsize=(8, 5))
            plt.plot(x, u_exact, 'k-', label='Exact')
            plt.plot(x, u_lf, 'b--', label='Lax-Friedrichs')
            plt.plot(x, u_ri, 'r--', label='Richtmyer')
            plt.plot(x, u_mc, 'g--', label='MacCormack')
            plt.title(f'Time = {t:.1f}')
            plt.xlabel('x')
            plt.ylabel('u')
            plt.ylim(-0.2, 2.5)
            plt.legend()
            plt.grid()
            plt.show()

solve_and_plot()
