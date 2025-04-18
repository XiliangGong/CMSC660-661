import numpy as np

n = 100  # 网格点数
xi = np.linspace(-1, 1, n + 2)  # 包含边界点，长度 n+2
h = xi[1] - xi[0]


def initial_condition_1(xi):
    # u(x, 0) = 1 - x^2, |x| < 1
    return np.where(np.abs(xi) < 1, 1 - xi**2, 0.0)

def initial_condition_2(xi):
    # u(x, 0) = 1 - 0.99*cos(2πx), |x| < 1
    return np.where(np.abs(xi) < 1, 1 - 0.99*np.cos(2*np.pi*xi), 0.0)


def spatial_derivatives(u, h):
    dudxi = np.zeros_like(u)
    d2udxi2 = np.zeros_like(u)

    # 内部节点：中心差分
    dudxi[1:-1] = (u[2:] - u[:-2]) / (2 * h)
    d2udxi2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / h**2

    # 边界节点：一边差分（二阶准确）
    dudxi[0] = ( -3*u[0] + 4*u[1] - u[2] ) / (2*h)
    dudxi[-1] = ( 3*u[-1] - 4*u[-2] + u[-3] ) / (2*h)
    d2udxi2[0] = d2udxi2[-1] = 0  # 可以设为0或用非对称差分近似

    return dudxi, d2udxi2

def rhs(t, u, xi, B=1.0, t0=1e-5):
    dudxi, d2udxi2 = spatial_derivatives(u, h)
    xft_inv_sq = (B**-2) * (t - t0)**(-2/3)
    
    term1 = -0.5 * ((1 + xi) * dudxi + (1 - xi) * dudxi) * dudxi
    term2 = u * d2udxi2
    term3 = dudxi**2

    return xft_inv_sq * (term1 + term2 + term3)

from scipy.integrate import solve_ivp

t_span = (0.1, 1.2)
t_eval = np.linspace(*t_span, 5)

# 初始条件选择：
u0 = initial_condition_2(xi)  # 或者 initial_condition_2(xi)

sol = solve_ivp(
    fun=lambda t, u: rhs(t, u, xi),
    t_span=t_span,
    y0=u0,
    t_eval=t_eval,
    method='RK45',  # 可以改成 stiff solver like 'BDF' if needed
)

import matplotlib.pyplot as plt

for i, t in enumerate(sol.t):
    u = sol.y[:, i]
    umax = np.max(u)
    plt.plot(xi, u / umax, label=f't={t:.2f}')

plt.plot(xi, 1 - xi**2, 'k--', label=r'$1 - \xi^2$')  # 自相似参考解
plt.xlabel(r'$\xi$')
plt.ylabel(r'$u(\xi, t)/u_{\max}(t)$')
plt.title('Self-similar collapse')
plt.legend()
plt.grid(True)
plt.show()
import matplotlib.pyplot as plt

for i, t in enumerate(sol.t):
    u = sol.y[:, i]
    umax = np.max(u)
    plt.plot(xi, u , label=f't={t:.2f}')
    

plt.plot(xi, 1 - xi**2, 'k--', label=r'$1 - \xi^2$')  # 自相似参考解
plt.xlabel(r'$\xi$')
plt.ylabel(r'$u(\xi, t)$')
plt.title('Self-similar collapse')
plt.legend()
plt.grid(True)
plt.show()
