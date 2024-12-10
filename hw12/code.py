import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from Levenberg_Marquardt import LevenbergMarquardt

# Utility Functions
def load_and_preprocess_data(file_path):
    """
    Load MNIST data from the .mat file and preprocess it.

    Args:
        file_path (str): Path to the .mat file.

    Returns:
        tuple: Preprocessed training and test data with their labels.
    """
    mnist_data = io.loadmat(file_path)
    
    imgs_train = mnist_data['imgs_train'].transpose(2, 0, 1)  # Convert to (samples, height, width)
    imgs_test = mnist_data['imgs_test'].transpose(2, 0, 1)
    labels_train = np.squeeze(mnist_data['labels_train'])
    labels_test = np.squeeze(mnist_data['labels_test'])

    # Filter by labels 1 and 7
    train_filter = np.isin(labels_train, [1, 7])
    test_filter = np.isin(labels_test, [1, 7])

    imgs_train_filtered = imgs_train[train_filter]
    labels_train_filtered = labels_train[train_filter]
    imgs_test_filtered = imgs_test[test_filter]
    labels_test_filtered = labels_test[test_filter]

    return imgs_train_filtered, labels_train_filtered, imgs_test_filtered, labels_test_filtered

def perform_pca(dataset, n_components):
    """
    Perform PCA on the dataset to reduce dimensionality.

    Args:
        dataset (ndarray): Input dataset with shape (samples, height, width).
        n_components (int): Number of principal components to retain.

    Returns:
        ndarray: Transformed dataset after PCA.
    """
    data_flat = dataset.reshape(dataset.shape[0], -1)  # Flatten each image
    print(f"Data shape before PCA: {data_flat.shape}")

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_flat)

    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)
    print(f"Data shape after PCA: {data_pca.shape}")

    return data_pca

def logloss_quadratic(X, y, w):
    return 0.5 * np.sum((np.log(1. + np.exp(-myquadratic(X, y, w))))**2)

def Res_and_Jac(X, y, w):
    aux = np.exp(-myquadratic(X, y, w))
    r = np.log(1. + aux)
    a = -aux / (1. + aux)
    n, d = X.shape
    d2 = d * d
    ya = y * a
    qterm = np.zeros((n, d2))
    for k in range(n):
        xk = X[k, :]
        xx = np.outer(xk, xk)
        qterm[k, :] = np.reshape(xx, (np.size(xx),))
    J = np.concatenate((qterm, X, np.ones((n, 1))), axis=1)
    for k in range(n):
        J[k, :] *= ya[k]
    return r, J

def myquadratic(X, y, w):
    d = np.size(X, axis=1)
    d2 = d * d
    W = np.reshape(w[:d2], (d, d))
    v = w[d2:d2 + d]
    b = w[-1]
    qterm = np.diag(X @ W @ X.T)
    q = y * qterm + (np.outer(y, np.ones(d)) * X) @ v + y * b
    return q

def gauss_newton(r_and_J, w, iter_max, tol):
    Loss_vals = []
    gradnorm_vals = []
    for i in range(iter_max):
        r, J = r_and_J(w)
        JTJ = J.T @ J + np.eye(J.shape[1]) * 1e-4
        JTJ_inv = np.linalg.inv(JTJ)
        grad = J.T @ r
        delta_w = -JTJ_inv @ grad
        w += delta_w

        loss = 0.5 * np.sum(r**2)
        grad_norm = np.linalg.norm(grad)

        Loss_vals.append(loss)
        gradnorm_vals.append(grad_norm)

        if grad_norm < tol:
            break

    return w, i + 1, np.array(Loss_vals), np.array(gradnorm_vals)

def stochastic_gradient_descent(X, y, w, lam, batch_size, step_size, max_epochs, tol, step_size_decay=None):
    n = X.shape[0]
    Loss_vals = []
    gradnorm_vals = []

    for epoch in range(max_epochs):
        indices = np.arange(n)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            r, J = Res_and_Jac(X_batch, y_batch, w)
            grad = J.T @ r + lam * w
            w -= step_size * grad

            loss = new_loss_function(X_batch, y_batch, w, lam)
            grad_norm = np.linalg.norm(grad)

            Loss_vals.append(loss)
            gradnorm_vals.append(grad_norm)

            if grad_norm < tol:
                break

        if step_size_decay:
            step_size *= step_size_decay

        if grad_norm < tol:
            break

    return w, epoch + 1, np.array(Loss_vals), np.array(gradnorm_vals)

# Main Program
if __name__ == "__main__":
    FILE_PATH = "./mnist2.mat"
    N_COMPONENTS = 20

    # Load and preprocess data
    imgs_train, labels_train, imgs_test, labels_test = load_and_preprocess_data(FILE_PATH)

    print(f"Filtered training data shape: {imgs_train.shape}")
    print(f"Filtered training labels shape: {labels_train.shape}")
    print(f"Filtered test data shape: {imgs_test.shape}")
    print(f"Filtered test labels shape: {labels_test.shape}")

    # Perform PCA
    X_train = perform_pca(imgs_train, N_COMPONENTS)
    X_test = perform_pca(imgs_test, N_COMPONENTS)

    lbl_train = labels_train
    lbl_test = labels_test

    # Initial parameters
    d = N_COMPONENTS
    w = np.ones((d * d + d + 1,))
    iter_max = 600
    tol = 1e-3

    def r_and_J(w):
        return Res_and_Jac(X_train, lbl_train, w)

    # Train with Gauss-Newton
    w, Niter, Loss_vals, gradnorm_vals = gauss_newton(r_and_J, w, iter_max, tol)

    # Plot learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(Loss_vals, label="Loss")
    ax1.set_xlabel("Iteration #")
    ax1.set_ylabel("Loss Function")
    ax1.set_yscale("log")
    ax1.set_title(f"Loss, Gauss-Newton, N_COMPONENTS = {N_COMPONENTS}")
    ax1.legend()

    ax2.plot(gradnorm_vals, label="||grad Loss||")
    ax2.set_xlabel("Iteration #")
    ax2.set_ylabel("Gradient Norm")
    ax2.set_yscale("log")
    ax2.set_title(f"Gradient Norm, Gauss-Newton, N_COMPONENTS = {N_COMPONENTS}")
    ax2.legend()

    plt.tight_layout()
    plt.show()


    # Define range of PCA components to test
    pca_components = range(1, 70, 2)
    misclassified_counts = []

    for n_components in pca_components:
        X_train_pca = perform_pca(imgs_train, n_components)
        X_test_pca = perform_pca(imgs_test, n_components)

        w = np.ones((n_components * n_components + n_components + 1,))
        def r_and_J(w):
            return Res_and_Jac(X_train_pca, lbl_train, w)

        w, _, _, _ = LevenbergMarquardt(r_and_J, w, iter_max, tol)

        test = myquadratic(X_test_pca, lbl_test, w)
        misses = np.argwhere(test < 0)
        misclassified_counts.append(len(misses))

    # Plot PCA component results
    plt.figure(figsize=(10, 6))
    plt.plot(pca_components, misclassified_counts, marker='o')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Number of Misclassified Digits')
    plt.title('Misclassified Digits vs PCA Components')
    plt.grid(True)
    plt.show()




    batch_sizes = [16, 32, 128]
    step_sizes = [0.1, 0.01, 0.001]
    step_size_decays = [None, 0.99, 0.95]
    # Plot the results with better colors
    fig, axes = plt.subplots(len(batch_sizes), len(step_sizes), figsize=(15, 10), sharex=True, sharey=True)
    results = []
    max_epochs = 100
    lam = 0.1

    results = []

        # Ensure w and d are correctly initialized
    fig, axes = plt.subplots(len(batch_sizes), len(step_sizes), figsize=(18, 12), sharex=True, sharey=True)

    # Define a color palette for better distinction
    color_palette = plt.cm.Set2.colors

    for i, batch_size in enumerate(batch_sizes):
        for j, step_size in enumerate(step_sizes):
            ax = axes[i, j]
            for k, step_size_decay in enumerate(step_size_decays):
                for result in results:
                    if result[0] == batch_size and result[1] == step_size and result[2] == step_size_decay:
                        ax.plot(
                            result[4], 
                            label=f"Decay Rate: {step_size_decay}", 
                            color=color_palette[k % len(color_palette)],
                            linewidth=1.5
                        )
            ax.set_title(f"Batch Size: {batch_size}, Step Size: {step_size}", fontsize=12, pad=10)
            ax.set_yscale("log")
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.legend(fontsize=9, loc="best", title="Decay", title_fontsize=10)

    # Add overall figure title and axis labels
    fig.suptitle("Comparison of Loss Curves Across Batch Sizes and Step Sizes", fontsize=16, weight="bold")
    fig.text(0.5, 0.04, "Iterations", ha="center", fontsize=14, weight="bold")
    fig.text(0.04, 0.5, "Loss (Logarithmic Scale)", va="center", rotation="vertical", fontsize=14, weight="bold")

    # Adjust layout to minimize overlapping
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.show()

