import numpy as np
import matplotlib.pyplot as plt

# Define parameters
J = 10
h = 1 / J

# Define the meshgrid (excluding boundaries for Dirichlet conditions)
x = np.linspace(0, 1, J+1)[1:-1]  # Remove boundaries at 0 and 1
y = np.linspace(0, 1, J+1)[1:-1]
X, Y = np.meshgrid(x, y)

# Define the eigenvector function
def eigenvector(kx, ky):
    return np.sin(kx * X) * np.sin(ky * Y)

# Define the eigenvalue function for reference
def eigenvalue(kx, ky):
    return -4 / h**2 * (np.sin(kx * h / 2)**2) * (np.sin(ky * h / 2)**2)

# Define the selected eigenvectors for plotting
eigenvectors = [
    (np.pi, np.pi),                  # Smallest eigenvalue
    (np.pi, 2 * np.pi),              # Intermediate eigenvalue
    ((J - 1) * np.pi, (J - 1) * np.pi)  # Largest eigenvalue
]

# Plot each eigenvector
for kx, ky in eigenvectors:
    eig_val = eigenvalue(kx, ky)
    plt.figure()
    plt.contourf(X, Y, eigenvector(kx, ky), cmap='viridis')
    plt.title(f"Eigenvector with $k_x = {kx/np.pi}\pi$, $k_y = {ky/np.pi}\pi$\nEigenvalue = {eig_val:.4f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Amplitude")
    plt.show()
