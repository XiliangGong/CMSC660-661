import numpy as np
import matplotlib.pyplot as plt

# Define the Rosenbrock function and its gradient
def rosenbrock(x):
    """Compute the Rosenbrock function value."""
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    """Compute the gradient of the Rosenbrock function."""
    grad_x = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    grad_y = 200 * (x[1] - x[0]**2)
    return np.array([grad_x, grad_y])

def rosenbrock_hessian(x):
    """Compute the Hessian matrix of the Rosenbrock function."""
    hessian = np.array([[1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]],
                        [-400 * x[0], 200]])
    return hessian

# Trust-region BFGS algorithm with Dogleg solver
def trust_region_bfgs(x0, grad_func, max_iter=100, delta=1.0, tol_grad=1e-6, tol_f=1e-8):
    """
    Perform optimization using the trust-region BFGS algorithm.
    
    Args:
        x0 (array): Initial point.
        grad_func (function): Function to compute the gradient.
        max_iter (int): Maximum number of iterations.
        delta (float): Trust region radius.
        tol_grad (float): Gradient norm tolerance for convergence.
        tol_f (float): Function value change tolerance for convergence.
    
    Returns:
        all_x (array): Sequence of points visited during optimization.
    """
    x = x0.copy()
    n = len(x0)
    B = np.eye(n)  # Initial Hessian approximation
    all_x = [x.copy()]
    prev_f = rosenbrock(x)
    
    for _ in range(max_iter):
        grad = grad_func(x)
        if np.linalg.norm(grad) < tol_grad:
            break
        
        # Compute BFGS step direction
        p_bfgs = -np.linalg.inv(B).dot(grad)
        if np.linalg.norm(p_bfgs) <= delta:
            p = p_bfgs
        else:
            p = -delta * grad / np.linalg.norm(grad)  # Dogleg step
        
        # Update variables
        x_new = x + p
        current_f = rosenbrock(x_new)
        if abs(current_f - prev_f) < tol_f:
            break
        grad_new = grad_func(x_new)
        y = grad_new - grad
        s = x_new - x
        
        # Update Hessian approximation
        if np.dot(y, s) > 0:
            B += np.outer(y, y) / np.dot(y, s) - (B @ np.outer(s, s) @ B) / np.dot(s, B @ s)
        
        x, prev_f = x_new, current_f
        all_x.append(x.copy())
    
    return np.array(all_x)

# Trust-region Newton algorithm with exact subspace solver
def trust_region_newton(x0, grad_func, hessian_func, max_iter=100, delta=1.0, tol_grad=1e-6, tol_f=1e-8):
    """
    Perform optimization using the trust-region Newton algorithm.
    
    Args:
        x0 (array): Initial point.
        grad_func (function): Function to compute the gradient.
        hessian_func (function): Function to compute the Hessian.
        max_iter (int): Maximum number of iterations.
        delta (float): Trust region radius.
        tol_grad (float): Gradient norm tolerance for convergence.
        tol_f (float): Function value change tolerance for convergence.
    
    Returns:
        all_x (array): Sequence of points visited during optimization.
    """
    x = x0.copy()
    all_x = [x.copy()]
    prev_f = rosenbrock(x)
    
    for _ in range(max_iter):
        grad = grad_func(x)
        if np.linalg.norm(grad) < tol_grad:
            break
        
        hess = hessian_func(x)
        p_newton = -np.linalg.inv(hess).dot(grad)
        if np.linalg.norm(p_newton) <= delta:
            p = p_newton
        else:
            p = -delta * grad / np.linalg.norm(grad)  # Trust-region step
        
        x_new = x + p
        current_f = rosenbrock(x_new)
        if abs(current_f - prev_f) < tol_f:
            break
        x, prev_f = x_new, current_f
        all_x.append(x.copy())
    
    return np.array(all_x)

# Visualization functions
def plot_contours_and_paths(X, Y, Z, paths, labels, colors):
    """Plot the Rosenbrock function contours and optimization paths."""
    plt.figure(figsize=(12, 6))
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
    for path, label, color in zip(paths, labels, colors):
        plt.plot(path[:, 0], path[:, 1], marker='o', color=color, label=label)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Level Sets of the Rosenbrock Function with Optimization Paths')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()

def plot_convergence(paths, labels, colors, optimal_point=np.array([1, 1])):
    """Plot the convergence of optimization methods."""
    plt.figure(figsize=(12, 6))
    for path, label, color in zip(paths, labels, colors):
        distances = np.linalg.norm(path - optimal_point, axis=1)
        plt.plot(range(len(distances)), distances, marker='o', color=color, label=label)
    plt.yscale('log')
    plt.xlabel('Iteration k')
    plt.ylabel('|| (x_k, y_k) - (x*, y*) ||')
    plt.title('Convergence of Optimization Methods')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()


from mpl_toolkits.mplot3d import Axes3D

def plot_3d_surface_with_paths(X, Y, Z, paths, labels, colors):
    """
    Plot the Rosenbrock function as a 3D surface and overlay optimization paths.
    
    Args:
        X, Y, Z: Meshgrid and function values for the Rosenbrock function.
        paths: List of optimization paths (list of arrays).
        labels: List of labels for each path.
        colors: List of colors for each path.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('Rosenbrock Function with Optimization Paths')

    # Overlay optimization paths
    for path, label, color in zip(paths, labels, colors):
        z_values = [rosenbrock(point) for point in path]
        ax.plot(path[:, 0], path[:, 1], z_values, marker='o', color=color, label=label)

    ax.legend(loc='upper left')
    plt.show()


def plot_contours_for_init(X, Y, Z, paths, labels, colors, init_index):
    """
    Plot the Rosenbrock function contours and optimization paths for a specific initial point.

    Args:
        X, Y, Z: Meshgrid and function values for the Rosenbrock function.
        paths: List of optimization paths (list of arrays).
        labels: List of labels for each path.
        colors: List of colors for each path.
        init_index: Index of the initial point (0 for Init 1, 1 for Init 2).
    """
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.8)
    for i, (path, label, color) in enumerate(zip(paths, labels, colors)):
        if f"Init {init_index + 1}" in label:  # Filter paths for the current initial point
            plt.plot(path[:, 0], path[:, 1], marker='o', color=color, label=label)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Optimization Paths for Initial Point {init_index + 1}')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    # Generate grid for 2D and 3D visualizations
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = 100 * (Y - X**2)**2 + (1 - X)**2

    # Initial conditions and method configurations
    initial_conditions = [np.array([1.2, 1.2]), np.array([-1.2, 1])]
    methods = [
        (trust_region_bfgs, "Trust-Region BFGS"),
        (trust_region_newton, "Trust-Region Newton")
    ]
    
    # Define colors for init1 and init2 for each method
    method_colors = {
        "Trust-Region BFGS": ['tab:orange', 'tab:green'],
        "Trust-Region Newton": ['tab:blue', 'tab:purple']
    }
    
    # Run optimization and store paths
    all_paths = []
    all_labels = []
    all_colors = []
    for i, x0 in enumerate(initial_conditions):
        for method, label in methods:
            if method == trust_region_bfgs:
                path = method(x0, rosenbrock_grad)
            else:
                path = method(x0, rosenbrock_grad, rosenbrock_hessian)
            all_paths.append(path)
            all_labels.append(f"{label} (Init {i+1})")
            all_colors.append(method_colors[label][i])

    # Choose visualization type
    visualization_type = "2D"  # Options: "2D" or "3D"

    if visualization_type == "2D":
        # Plot separate 2D contour plots for Init 1 and Init 2
        plot_contours_for_init(X, Y, Z, all_paths, all_labels, all_colors, init_index=0)
        plot_contours_for_init(X, Y, Z, all_paths, all_labels, all_colors, init_index=1)
    elif visualization_type == "3D":
        # Plot 3D surface with optimization paths
        plot_3d_surface_with_paths(X, Y, Z, all_paths, all_labels, all_colors)
        plot_convergence(all_paths, all_labels, all_colors)
