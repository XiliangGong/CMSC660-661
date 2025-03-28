import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

n = 100
xmin = -np.pi
xmax = np.pi
ymin = 0
ymax = 2
hx = (xmax - xmin) / n
hy = (ymax - ymin) / n
x = np.linspace(xmin, xmax, n+1)  # n+1 points to include xmax
y = np.linspace(ymin, ymax, n+1)  # n+1 points to include ymax
xg, yg = np.meshgrid(x, y, indexing='ij')  # 'ij' for MATLAB-like indexing

e = np.ones(n)
# Periodic boundary in x-direction (Lx)
Lx = sp.diags([e, e, -2*e, e, e], [-n+1, -1, 0, 1, n-1], shape=(n, n), format='lil')
# Neumann boundary in y-direction (Ly)
Ly = sp.diags([e, -2*e, e], [-1, 0, 1], shape=(n, n), format='lil')
Ly[0, 1] = 2  # Neumann boundary condition at y=0

# Convert back to efficient format for computation
Lx = Lx.tocsr()
Ly = Ly.tocsr()

# Create f matrix
f = np.zeros((n, n))
for i in range(n):
    xi = x[i]
    if -np.pi/2 <= xi <= np.pi/2:
        f[i, :] = -np.cos(xi)
f = f.reshape(n*n, 1)

I = sp.eye(n, format='csr')
A = (1/hx**2) * sp.kron(I, Lx) + (1/hy**2) * sp.kron(Ly, I)

# Solve the system
u = spla.spsolve(A, f.flatten())
u = u.reshape(n, n)

# Pad with periodic boundary in x and Neumann in y
u_full = np.zeros((n+1, n+1))
u_full[:n, :n] = u
u_full[n, :n] = u_full[0, :n]  # Periodic in x

# Create extended grid for plotting
x_plot = np.linspace(xmin, xmax, n+1)
y_plot = np.linspace(ymin, ymax, n+1)
xg_plot, yg_plot = np.meshgrid(x_plot, y_plot, indexing='ij')

# Plotting
plt.figure(figsize=(10, 6))
contour = plt.contourf(xg_plot, yg_plot, u_full, levels=20)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.colorbar(contour)
plt.title('Solution Contour Plot')
plt.show()
