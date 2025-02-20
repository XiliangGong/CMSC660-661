import numpy as np
import matplotlib.pyplot as plt

# 定义复平面网格
x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y  # 复数平面

# 定义不同显式Runge-Kutta方法的稳定性函数
def stability_forward_euler(z):
    return 1 + z

def stability_midpoint_euler(z):
    return 1 + z + 0.5 * z**2

def stability_kutta_3rd(z):
    return 1 + z + (1/2) * z**2 + (1/6) * z**3

def stability_rk4(z):
    return 1 + z + (1/2) * z**2 + (1/6) * z**3 + (1/24) * z**4

def stability_dopri5(z):
    return 1 + z + (1/2) * z**2 + (1/6) * z**3 + (1/24) * z**4 + (1/120) * z**5

# 设定不同的显式Runge-Kutta方法
methods = {
    "Forward Euler": stability_forward_euler,
    "Midpoint Euler": stability_midpoint_euler,
    "Kutta 3rd Order": stability_kutta_3rd,
    "RK4 (4th Order)": stability_rk4,
    "DOPRI5 (5th Order)": stability_dopri5
}

# 颜色列表
colors = ['b', 'g', 'r', 'c', 'm']

# 创建子图，每个方法单独绘制
plt.figure(figsize=(12, 10))

for idx, (name, stability_function) in enumerate(methods.items(), start=1):
    plt.subplot(3, 2, idx)  # 3行2列的子图布局
    R = np.abs(stability_function(Z))  # 计算 |R(z)|
    plt.contour(X, Y, R, levels=[1], colors=[colors[idx - 1]], linewidths=1.5)
    
    # 图像格式调整
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title(f"Stability Region of {name}")
    plt.grid()

# 调整子图间距
plt.tight_layout()
plt.show()
