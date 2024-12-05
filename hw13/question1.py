from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 加载MNIST数据集
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

# 提取标签为1和7的数据
mask = (y == '1') | (y == '7')
X, y = X[mask], y[mask]

# 将标签转换为二分类
y = np.where(y == '1', 1, -1)

# 标准化特征数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def loss_function(w, X, y, lambd):
    n = X.shape[0]
    q = y * (X @ w[:-1] + w[-1])  # q(x_j; w)
    return np.mean(np.log(1 + np.exp(-q))) + (lambd / 2) * np.linalg.norm(w)**2

def gradient(w, X, y, lambd):
    n = X.shape[0]
    q = y * (X @ w[:-1] + w[-1])  # q(x_j; w)
    grad = -X.T @ (y / (1 + np.exp(q))) / n + lambd * w[:-1]  # W和v的梯度
    grad_b = -np.mean(y / (1 + np.exp(q)))  # 偏置b的梯度
    return np.concatenate([grad, [grad_b]])  # 合并成完整梯度


def nesterov_optimizer(w, grad_func, X, y, lambd, eta, beta, num_epochs, batch_size=None):
    v = np.zeros_like(w)
    n = X.shape[0]
    loss_history = []  # 用于记录损失值
    for epoch in range(num_epochs):
        if batch_size is None:  # 确定性版本
            grad = grad_func(w + beta * v, X, y, lambd)
        else:  # 随机版本
            indices = np.random.choice(n, batch_size, replace=False)
            grad = grad_func(w + beta * v, X[indices], y[indices], lambd)
        v = beta * v - eta * grad
        w += v
        # 记录损失值
        loss = loss_function(w, X, y, lambd)
        loss_history.append(loss)
    return w, loss_history

def adam_optimizer(w, grad_func, X, y, lambd, eta, beta1, beta2, epsilon, num_epochs, batch_size=None):
    m, v = np.zeros_like(w), np.zeros_like(w)
    n = X.shape[0]
    t = 0
    loss_history = []  # 用于记录损失值
    for epoch in range(num_epochs):
        if batch_size is None:  # 确定性版本
            grad = grad_func(w, X, y, lambd)
        else:  # 随机版本
            indices = np.random.choice(n, batch_size, replace=False)
            grad = grad_func(w, X[indices], y[indices], lambd)
        
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w -= eta * m_hat / (np.sqrt(v_hat) + epsilon)
        # 记录损失值
        loss = loss_function(w, X, y, lambd)
        loss_history.append(loss)
    return w, loss_history

# 初始化参数
d = X_train.shape[1]
w_init = np.zeros(d + 1)  # 包含偏置b
lambd = 0.1  # 正则化参数
eta = 0.01  # 学习率
num_epochs = 50
batch_size = 64


# 确定性和随机Nesterov
w_nesterov, loss_nesterov = nesterov_optimizer(w_init, gradient, X_train, y_train, lambd, eta, beta=0.9, num_epochs=num_epochs)
w_nesterov_stochastic, loss_nesterov_stochastic = nesterov_optimizer(w_init, gradient, X_train, y_train, lambd, eta, beta=0.9, num_epochs=num_epochs, batch_size=batch_size)

# 确定性和随机Adam
w_adam, loss_adam = adam_optimizer(w_init, gradient, X_train, y_train, lambd, eta, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=num_epochs)
w_adam_stochastic, loss_adam_stochastic = adam_optimizer(w_init, gradient, X_train, y_train, lambd, eta, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=num_epochs, batch_size=batch_size)

# 评估
print("Nesterov Loss:", loss_function(w_nesterov, X_test, y_test, lambd))
print("Stochastic Nesterov Loss:", loss_function(w_nesterov_stochastic, X_test, y_test, lambd))
print("Adam Loss:", loss_function(w_adam, X_test, y_test, lambd))
print("Stochastic Adam Loss:", loss_function(w_adam_stochastic, X_test, y_test, lambd))

import matplotlib.pyplot as plt

epochs = np.arange(1, num_epochs + 1)

plt.plot(epochs, loss_nesterov, label='Nesterov')
plt.plot(epochs, loss_nesterov_stochastic, label='Stochastic Nesterov')
plt.plot(epochs, loss_adam, label='Adam')
plt.plot(epochs, loss_adam_stochastic, label='Stochastic Adam')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs Epochs')
plt.savefig('loss_vs_epochs.png')
plt.show()

