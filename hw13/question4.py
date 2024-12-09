import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数等值线
def objective_function(x, y):
    return (x - 1)**2 + (y - 2.5)**2

# 定义约束
constraints = [
    lambda x, y: x - 2*y + 2,  # 约束1
    lambda x, y: -x - 2*y + 6,  # 约束2
    lambda x, y: -x + 2*y + 2,  # 约束3
    lambda x, y: x,  # 约束4
    lambda x, y: y   # 约束5
]

# 绘制可行域
x = np.linspace(-1, 5, 400)
y = np.linspace(-1, 5, 400)
X, Y = np.meshgrid(x, y)
Z = objective_function(X, Y)

# 创建图形
plt.figure(figsize=(8, 8))
plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)  # 绘制目标函数等值线

# 绘制约束区域
feasible_region = np.ones_like(X, dtype=bool)
for constraint in constraints:
    feasible_region &= (constraint(X, Y) >= 0)

plt.contourf(X, Y, feasible_region, levels=1, colors=['#d1e5f0'], alpha=0.6)

# 绘制可行域的边界
for constraint in constraints:
    plt.contour(X, Y, constraint(X, Y), levels=[0], colors='red', linestyles='dashed', linewidths=1.5)

# 标注点 (1, 2.5)（目标函数最小值点）
plt.scatter(1, 2.5, color='blue', label='Center of Objective Function (1, 2.5)', zorder=5)

# 图形设置
plt.title('Level Sets of Objective Function and Feasible Region')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()
plt.grid(alpha=0.5)


plt.show()

from scipy.optimize import minimize

# 定义目标函数
def objective(x):
    return (x[0] - 1)**2 + (x[1] - 2.5)**2

# 定义约束条件
constraints = [
    {'type': 'ineq', 'fun': lambda x: x[0] - 2*x[1] + 2},  # 约束1
    {'type': 'ineq', 'fun': lambda x: -x[0] - 2*x[1] + 6}, # 约束2
    {'type': 'ineq', 'fun': lambda x: -x[0] + 2*x[1] + 2}, # 约束3
    {'type': 'ineq', 'fun': lambda x: x[0]},               # 约束4 (x >= 0)
    {'type': 'ineq', 'fun': lambda x: x[1]}                # 约束5 (y >= 0)
]

# 设置初始猜测值
x0 = np.array([2, 0])

# 使用优化器求解问题
result = minimize(objective, x0, constraints=constraints, method='SLSQP')

# 打印结果
optimal_point = result.x
optimal_value = result.fun

print(optimal_point, optimal_value)

from scipy.optimize import linprog

# 定义梯度和约束
def gradient(x):
    return np.array([2 * (x[0] - 1), 2 * (x[1] - 2.5)])

# 活动集的初始约束 (对应约束 3 和 5)
active_constraints = [lambda x: -x[0] + 2*x[1] + 2, lambda x: x[1]]

# 定义活动约束的梯度矩阵
def active_constraint_matrix(x, active_indices):
    A = []
    for idx in active_indices:
        if idx == 3:
            A.append([-1, 2])  # 约束 3 的梯度
        elif idx == 5:
            A.append([0, 1])   # 约束 5 的梯度
    return np.array(A)

# 初始化
x_k = np.array([2.0, 0.0])  # 初始点
active_indices = [3, 5]  # 活动约束集

# 存储每次迭代的结果
iterations = []
iterations.append((x_k.copy(), active_indices))

# 最大迭代次数
max_iterations = 5

for _ in range(max_iterations):
    # 计算当前活动约束的矩阵和目标梯度
    A = active_constraint_matrix(x_k, active_indices)
    grad = gradient(x_k)
    
    # 解 KKT 系统 (求解最小二乘问题)
    # Minimize ||grad + A.T * λ|| subject to A * dx = 0
    if A.size > 0:
        dx = np.linalg.lstsq(A.T, -grad, rcond=None)[0]
        p_k = np.linalg.lstsq(A, dx, rcond=None)[0]  # 拉格朗日乘子
    else:
        dx = -grad
        p_k = None

    # 如果 dx 是零向量，则当前点是 KKT 点，停止迭代
    if np.allclose(dx, 0):
        break

    # 检查拉格朗日乘子是否有负值
    min_lambda_index = None
    if p_k is not None:
        for i, lambda_i in enumerate(p_k):
            if lambda_i < 0:
                min_lambda_index = i
                break

    # 如果存在负的拉格朗日乘子，移除对应的活动约束
    if min_lambda_index is not None:
        active_indices.pop(min_lambda_index)
    else:
        # 沿着方向 dx 移动，检查非活动约束是否变为等式
        step_size = 1.0
        new_constraint = None
        for i, constraint in enumerate(constraints):
            if i + 1 not in active_indices:
                # 检查约束是否变为等式
                value = constraint(x_k + step_size * dx)
                if value < 0:
                    step_size = min(step_size, -constraint(x_k) / (constraint(x_k + step_size * dx) - constraint(x_k)))
                    new_constraint = i + 1

        # 更新点 x_k
        x_k += step_size * dx

        # 添加新活动约束
        if new_constraint is not None:
            active_indices.append(new_constraint)

    # 保存迭代信息
    iterations.append((x_k.copy(), active_indices))

# 打印每次迭代的结果
iterations

# 修正活动集方法的迭代实现
for _ in range(max_iterations):
    # 计算当前活动约束的矩阵和目标梯度
    A = active_constraint_matrix(x_k, active_indices)
    grad = gradient(x_k)
    
    # 解 KKT 系统 (求解最小二乘问题)
    if A.size > 0:
        dx = np.linalg.lstsq(A.T, -grad, rcond=None)[0]
        p_k = np.linalg.lstsq(A, dx, rcond=None)[0]  # 拉格朗日乘子
    else:
        dx = -grad
        p_k = None

    # 如果 dx 是零向量，则当前点是 KKT 点，停止迭代
    if np.allclose(dx, 0):
        break

    # 检查拉格朗日乘子是否有负值
    min_lambda_index = None
    if p_k is not None:
        for i, lambda_i in enumerate(p_k):
            if lambda_i < 0:
                min_lambda_index = i
                break

    # 如果存在负的拉格朗日乘子，移除对应的活动约束
    if min_lambda_index is not None:
        active_indices.pop(min_lambda_index)
    else:
        # 沿着方向 dx 移动，检查非活动约束是否变为等式
        step_size = 1.0
        new_constraint = None
        for i, constraint in enumerate(constraints):
            if i + 1 not in active_indices:
                # 检查约束是否变为等式
                value = constraint['fun'](x_k + step_size * dx)
                if value < 0:
                    step_size = min(step_size, -constraint['fun'](x_k) / (constraint['fun'](x_k + step_size * dx) - constraint['fun'](x_k)))
                    new_constraint = i + 1

        # 更新点 x_k
        x_k += step_size * dx

        # 添加新活动约束
        if new_constraint is not None:
            active_indices.append(new_constraint)

    # 保存迭代信息
    iterations.append((x_k.copy(), active_indices))

# 打印每次迭代的结果
iterations



