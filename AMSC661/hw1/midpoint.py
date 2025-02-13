import numpy as np
import matplotlib.pyplot as plt

# Define the system of ODEs
def gravity_system(t, state):
    x, y, u, v = state
    r2 = x**2 + y**2  # Compute r^2
    return np.array([u, v, -x/r2, -y/r2])  # Returns [dx/dt, dy/dt, du/dt, dv/dt]

# Midpoint rule with Forward Euler predictor
def midpoint_method(N, T_final):
    dt = 2 * np.pi / N  # Step size
    steps = int(T_final / dt)  # Number of steps
    state = np.zeros((steps + 1, 4))  # Store [x, y, u, v] for all steps

    # Initial conditions
    state[0] = np.array([1.0, 0.0, 0.0, 1.0])

    for n in range(steps):
        # Forward Euler predictor
        predictor = state[n] + dt * gravity_system(n * dt, state[n])
        # Midpoint correction
        state[n + 1] = state[n] + dt * gravity_system((n + 0.5) * dt, 0.5 * (state[n] + predictor))

    return state

# Define parameters
T_final = 8 * np.pi  # Time interval
N_values = [20, 40, 80]  # Different step sizes

# Solve and plot for different step sizes
plt.figure(figsize=(10, 6))

for N in N_values:
    sol = midpoint_method(N, T_final)


    # Compute norm at final time (显示为指数格式)
    final_norm = np.linalg.norm(sol[-1, :2])  # Norm of (x, y)
    print(f"N = {N}, Norm at t = 4π: {final_norm:.2e}")  # 改成指数格式
    # Plot trajectory
    plt.plot(sol[:, 0], sol[:, 1], label=f"N = {N}")

# Configure and show the plot
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Gravity Problem: Midpoint Rule with Forward Euler Predictor")
plt.legend()
plt.grid()
plt.show()
