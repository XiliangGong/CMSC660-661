import numpy as np
import matplotlib.pyplot as plt

def acceleration(x, y):
    r = np.sqrt(x**2 + y**2)
    return -x / r**3, -y / r**3

def stoermer_verlet(u0, v0, x0, y0, T, steps_per_period, num_periods):
    h = T / steps_per_period
    steps = steps_per_period * num_periods
    
    x = np.zeros(steps)
    y = np.zeros(steps)
    u = np.zeros(steps)
    v = np.zeros(steps)
    H = np.zeros(steps)

    x[0], y[0] = x0, y0
    u[0], v[0] = u0, v0

    # Initial half-step for momentum
    ax, ay = acceleration(x[0], y[0])
    u_half = u[0] + 0.5 * h * ax
    v_half = v[0] + 0.5 * h * ay

    for i in range(1, steps):
        # Update positions
        x[i] = x[i-1] + h * u_half
        y[i] = y[i-1] + h * v_half

        # Compute acceleration
        ax, ay = acceleration(x[i], y[i])

        # Full step for momentum
        u_half += h * ax
        v_half += h * ay

        # Store full momentum
        u[i] = u_half - 0.5 * h * ax
        v[i] = v_half - 0.5 * h * ay

        # Compute Hamiltonian
        r = np.sqrt(x[i]**2 + y[i]**2)
        H[i] = 0.5 * (u[i]**2 + v[i]**2) - 1 / r

    return x, y, H

# Parameters
u0, v0 = 0, 0.5
x0, y0 = 2, 0
T = 2 * np.pi * (4/3)**(3/2)
steps_per_period = 100
num_periods = 10

# Run simulation
x, y, H = stoermer_verlet(u0, v0, x0, y0, T, steps_per_period, num_periods)

# Plot trajectory
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Elliptic Trajectory')
plt.grid()
plt.legend()
plt.axis('equal')
plt.show()

# Plot Hamiltonian over time
plt.figure(figsize=(8, 6))
plt.plot(H, label='Hamiltonian')
plt.xlabel('Step')
plt.ylabel('Hamiltonian')
plt.title('Hamiltonian vs Time')
plt.grid()
plt.legend()
plt.show()
