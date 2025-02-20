import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

# Define the Arenstorf orbit equations
def arenstorf_orbit(t, y):
    mu = 0.012277471
    mu_hat = 1 - mu

    x, y_, vx, vy = y
    r1 = np.sqrt((x + mu)**2 + y_**2)
    r2 = np.sqrt((x - mu_hat)**2 + y_**2)

    dxdt = vx
    dydt = vy
    dvxdt = x + 2*vy - mu_hat*(x + mu)/r1**3 - mu*(x - mu_hat)/r2**3
    dvydt = y_ - 2*vx - mu_hat*y_/r1**3 - mu*y_/r2**3

    return [dxdt, dydt, dvxdt, dvydt]

# Initial conditions for the Arenstorf orbit
y0_arenstorf = [0.994, 0, 0, -2.00158510637908252240537862224]

# Time parameters
Tmax = 100
t_span_arenstorf = (0, Tmax)
t_eval_arenstorf = np.linspace(0, Tmax, 5000)

# Solvers to compare
solvers = ['RK45', 'DOP853', 'Radau']
cpu_times = {}

plt.figure(figsize=(8, 6))

for solver in solvers:
    start_time = time.time()
    sol = solve_ivp(arenstorf_orbit, t_span_arenstorf, y0_arenstorf, 
                    method=solver, t_eval=t_eval_arenstorf, atol=1e-12, rtol=1e-12)
    end_time = time.time()
    cpu_times[solver] = end_time - start_time
    
    # Plot the orbit
    plt.plot(sol.y[0], sol.y[1], label=f"{solver} (Time: {cpu_times[solver]:.2f}s)")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Arenstorf Periodic Orbit (x vs y)")
plt.legend()
plt.grid()
plt.show()

# Print CPU times
for solver, time_taken in cpu_times.items():
    print(f"Solver: {solver}, CPU Time: {time_taken:.4f} seconds")
