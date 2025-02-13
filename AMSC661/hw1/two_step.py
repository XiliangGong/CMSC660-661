import numpy as np
import matplotlib.pyplot as plt

# Define the system of ODEs
def gravity_system(t, state):
    x, y, u, v = state
    r2 = x**2 + y**2  # Compute r^2
    return np.array([u, v, -x/r2, -y/r2])  # Returns [dx/dt, dy/dt, du/dt, dv/dt]

# Explicit two-step method
def solve_gravity(N, T_final):
    dt = 2 * np.pi / N  # Compute step size
    steps = int(T_final / dt)  # Number of steps
    state = np.zeros((steps + 1, 4))  # Store [x, y, u, v] for all steps

    # Initial conditions
    state[0] = np.array([1.0, 0.0, 0.0, 1.0])

    # First step: use Euler’s method to get second value
    state[1] = state[0] + dt * gravity_system(0, state[0])

    # Apply the two-step explicit method
    for n in range(1, steps):
        state[n + 1] = -4 * state[n] + 5 * state[n - 1] + dt * (4 * gravity_system(n * dt, state[n]) + 2 * gravity_system((n-1) * dt, state[n-1]))

    return state

# Define parameters
T_final = 4 * np.pi  # Final time
N_values = [20, 40, 80]  # Different step sizes

# Solve and plot for different step sizes
plt.figure(figsize=(8, 6))

for N in N_values:
    sol = solve_gravity(N, T_final)

    # Compute norm at final time
    final_norm = np.linalg.norm(sol[-1, :2])  # Norm of (x, y)
    print(f"Final norm for N = {N}: {final_norm:.4e}")

    # Plot trajectory
    plt.plot(sol[:, 0], sol[:, 1], label=f"N = {N}")

# Configure and show the plot
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Gravity Problem: Numerical Trajectory")
plt.legend()
plt.grid()
plt.show()
