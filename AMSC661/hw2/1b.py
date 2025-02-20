import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

# Define the Van der Pol oscillator system
def van_der_pol(t, y, mu):
    y1, y2 = y
    dydt = [y2, mu * ((1 - y1**2) * y2) - y1]
    return dydt

# Parameters
mu_values = [10, 100, 1000]  # Increasing stiffness
tmax = 1000.0
t_span = (0, tmax)
t_eval = np.linspace(0, tmax, 1000)  # Reduce evaluation points for speed
tol_values = [1e-6, 1e-9, 1e-12]  # Error tolerances
initial_conditions = [2, 0]

# Store results
cpu_times = {solver: {mu: [] for mu in mu_values} for solver in ['RK45', 'LSODA']}

# Solve and plot
for mu in mu_values:
    plt.figure(figsize=(6, 4))
    for solver in ['RK45', 'LSODA']:
        for tol in tol_values:
            start_time = time.time()
            sol = solve_ivp(van_der_pol, t_span, initial_conditions, method=solver, 
                            args=(mu,), t_eval=t_eval, atol=tol, rtol=tol)
            end_time = time.time()
            cpu_time = end_time - start_time
            cpu_times[solver][mu].append(cpu_time)
        
        # Plot phase space trajectory (y1, y2)
        print(f"{solver}, mu={mu}: {sol.message}")
        plt.plot(sol.y[0], sol.y[1], label=f"{solver}, mu={mu}")
    
    plt.xlabel("$y_1$")
    plt.ylabel("$y_2$")
    plt.title(f"Van der Pol Oscillator Phase Space (mu={mu})")
    plt.legend()
    plt.grid()
    plt.show()

# Log-log plot of CPU time vs. error tolerance
plt.figure(figsize=(8, 6))
for solver in ['RK45', 'LSODA']:
    for mu in mu_values:
        plt.plot(np.log10(tol_values), np.log10(cpu_times[solver][mu]), 
                 marker='o', label=f"{solver}, mu={mu}")

plt.xlabel("log10(Tolerance)")
plt.ylabel("log10(CPU Time)")
plt.title("CPU Time vs. Tolerance for Van der Pol Oscillator")
plt.legend()
plt.grid()
plt.show()
