import numpy as np
import matplotlib.pyplot as plt

# Parameters
sigma0 = 0.5
k0 = 5
N = 512
L = 10.0
x = np.linspace(-L, L, N, endpoint=False)
dx = x[1] - x[0]
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
t_values = [0.0, 0.25, 0.5, 1.0]

# === 1. Analytical solution ===
def psi_analytic(x, t, sigma0, k0):
    A = (1 / (2 * np.pi * sigma0 ** 2))**0.25
    denom = 1 + 1j * t / (2 * sigma0**2)
    prefactor = A / np.sqrt(denom)
    phase = -((x - k0 * t)**2) / (4 * sigma0**2 * denom) + 1j * (k0 * x - 0.5 * k0**2 * t)
    return prefactor * np.exp(phase)

psis_analytic = [psi_analytic(x, t, sigma0, k0) for t in t_values]

# === 2. FFT-based time evolution ===
psi0 = (1 / (2 * np.pi * sigma0**2)**0.25) * np.exp(-x**2 / (4 * sigma0**2) + 1j * k0 * x)

def evolve_fft(psi0, k, t):
    psi_hat0 = np.fft.fft(psi0)
    phase = np.exp(-1j * k**2 * t / 2)
    return np.fft.ifft(psi_hat0 * phase)

psis_fft = [evolve_fft(psi0, k, t) for t in t_values]

# === 3. Spectral RK4 ===
psi = psi0.copy()
psis_saved_spectral = []

def L_psi_spectral(psi):
    psi_hat = np.fft.fft(psi)
    return 1j / 2 * np.fft.ifft(-(k**2) * psi_hat)

dt = 1e-4
save_steps = [int(t / dt) for t in t_values]
Nt = max(save_steps)

for step in range(Nt + 1):
    if step in save_steps:
        psis_saved_spectral.append(psi.copy())
    k1 = dt * L_psi_spectral(psi)
    k2 = dt * L_psi_spectral(psi + 0.5 * k1)
    k3 = dt * L_psi_spectral(psi + 0.5 * k2)
    k4 = dt * L_psi_spectral(psi + k3)
    psi += (k1 + 2 * k2 + 2 * k3 + k4) / 6

# === 4. Finite Difference RK4 ===
psi = psi0.copy()
psis_saved_fd = []

def L_psi_fd(psi, dx):
    d2psi = (np.roll(psi, -1) + np.roll(psi, 1) - 2 * psi) / dx**2
    return 1j / 2 * d2psi

for step in range(Nt + 1):
    if step in save_steps:
        psis_saved_fd.append(psi.copy())
    k1 = dt * L_psi_fd(psi, dx)
    k2 = dt * L_psi_fd(psi + 0.5 * k1, dx)
    k3 = dt * L_psi_fd(psi + 0.5 * k2, dx)
    k4 = dt * L_psi_fd(psi + k3, dx)
    psi += (k1 + 2 * k2 + 2 * k3 + k4) / 6

# === Plot all 4 methods in 2×2 subplot ===
methods = ["Analytical", "FFT Exact", "Spectral RK4", "Finite-Diff RK4"]
results = [psis_analytic, psis_fft, psis_saved_spectral, psis_saved_fd]

fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
for ax, result, title in zip(axs.flat, results, methods):
    for psi_t, t in zip(result, t_values):
        ax.plot(x, np.abs(psi_t)**2, label=f't={t}')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel(r'$|\psi(x,t)|^2$')
    ax.legend()
    ax.grid(True)

plt.suptitle(f'Comparison of 1D Schrödinger Equation Solutions (k₀ = {k0}, σ₀ = {sigma0} N = {N})', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
