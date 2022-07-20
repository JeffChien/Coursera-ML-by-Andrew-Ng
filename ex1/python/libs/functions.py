# %%
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

#%%
def patch_c0(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


# %%
def warm_up_exercise():
    return np.eye(5)


# %%
def plot_data(X, y):
    fig, ax = plt.subplots()
    ax.plot(X, y, "rx", markersize=10)
    return ax


# %%
def compute_cost(X: npt.NDArray, y: npt.NDArray, theta: npt.NDArray) -> np.float64:
    m = X.shape[0]
    error = (theta @ X.T) - y
    return (error @ error.T) / (2 * m)


# %%
def gradient_descent(X, y, theta, alpha, num_iters):
    m = X.shape[0]
    j_history = np.zeros(num_iters)
    for i in range(num_iters):
        theta = theta - (alpha / m) * (((theta @ X.T) - y) @ X)
        j_history[i] = compute_cost(X, y, theta)

    return theta, j_history


#%%
def mini_batch_gradient_descent(X, y, theta, alpha, num_iters, batch_size=10):
    m = X.shape[0]
    j_history = np.empty(0)
    theta_history = np.empty((0, 2))
    for i in range(num_iters):
        for start in range(0, m, batch_size):
            sample_X = X[start : start + batch_size]
            sample_y = y[start : start + batch_size]
            sample_m = X.shape[0]
            new_theta = theta - (alpha / sample_m) * (
                ((theta @ sample_X.T) - sample_y) @ sample_X
            )
            J = compute_cost(sample_X, sample_y, new_theta)
            j_history = np.append(j_history, J)
            theta_history = np.vstack((theta_history, new_theta))
            theta = theta_history[-1]
    return theta_history, j_history


# %%
def feature_normalize(X: npt.NDArray):
    mu = X.mean(0)
    X_norm = X - mu
    sigma = X_norm.std(0)
    return X_norm / sigma, mu, sigma


# %%
def normal_eqn(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y.T
