#%%
import math
import os
import sys
import pathlib
import itertools as it
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio

#%%
def plot_feature_hist(X, ncols=3, nbins=25):
    nfeatures = X.shape[1]
    ncols = min(ncols, nfeatures)
    nrows = math.ceil(nfeatures / ncols)

    fig, axs = plt.subplots(nrows, ncols)
    axs = axs.ravel()
    for i in range(nfeatures):
        axs[i].hist(X[:, i], nbins, edgecolor="black")
        axs[i].set_title(f"feature x{i}")
    return axs


def estimate_gaussian(X):
    m, n = X.shape

    mu = X.mean(0)
    sigma2 = ((X - mu) ** 2).sum(0) / m
    return mu, sigma2


#%%
def estimate_gaussian_density(X, mu, sigma2):
    n = mu.size
    X = X - mu

    p = np.prod(
        (2 * np.pi) ** (-0.5) * sigma2 ** (-0.5) * np.exp(-(X * X) / (2 * sigma2)), 1
    )
    return p


def visualize_fit(X, mu, sigma2, density_fn):
    X1, X2 = np.meshgrid(np.arange(0, 36, 0.5), np.arange(0, 36, 0.5))
    Z = density_fn(np.vstack((X1.ravel(), X2.ravel())).T, mu, sigma2)
    Z = Z.reshape(X1.shape)

    fig, ax = plt.subplots()
    ax.plot(X[:, 0], X[:, 1], "bx")
    if np.isinf(Z).sum() == 0:
        ax.contour(X1, X2, Z, 10.0 ** np.arange(-20, 0, 3))
    return ax


#%%
def select_threshold(yval, pval):
    best_eps = 0
    best_f1 = 0
    cur_f1 = 0

    step = (pval.max() - pval.min()) / 1000
    nums = int((pval.max() - pval.min()) / step)
    for eps in np.linspace(pval.min(), pval.max(), nums):
        pred = pval < eps
        tp = ((pred == 1) & (yval == 1)).sum()
        fp = ((pred == 1) & (yval == 0)).sum()
        fn = ((pred == 0) & (yval == 1)).sum()
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        cur_f1 = (2 * prec * rec) / (prec + rec) if prec + rec else 0
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_eps = eps
    return best_eps, best_f1


#%%
def estimate_multivariate_gaussian_density(X, mu, sigma2):
    n = mu.size
    Sigma2 = np.empty((n, n))
    if len(sigma2.shape) == 1 or sigma2.shape[1] == 1:
        Sigma2 = np.diag(sigma2)

    X = X - mu.T
    p = (
        (2 * np.pi) ** (-n / 2)
        * np.linalg.det(Sigma2) ** (-0.5)
        * np.exp(-0.5 * (X @ np.linalg.pinv(Sigma2) * X).sum(1))
    )
    return p


#%%
def cofi_cost_fn(params, Y, R, num_users, num_movies, num_features, lb):
    # X = m x n
    # Theta = u x n
    # Y = m x u
    # R = m x u
    X = params[: num_movies * num_features].reshape((num_movies, num_features))
    Theta = params[num_movies * num_features :].reshape((num_users, num_features))

    J = 0.5 * ((X @ Theta.T - Y) ** 2 * R).sum()
    reg = 0.5 * lb * ((Theta**2).sum() + (X**2).sum())

    return J + reg


#%%
def cofi_grad_fn(params, Y, R, num_users, num_movies, num_features, lb):
    # X = m x n
    # Theta = u x n
    # Y = m x u
    # R = m x u
    X = params[: num_movies * num_features].reshape((num_movies, num_features))
    Theta = params[num_movies * num_features :].reshape((num_users, num_features))

    common = (X @ Theta.T - Y) * R

    X_grad = common @ Theta + lb * X
    Theta_grad = common.T @ X + lb * Theta

    return np.hstack((X_grad.ravel(), Theta_grad.ravel()))


#%%
def cofi_cost_n_grad_fn(params, Y, R, num_users, num_movies, num_features, lb):
    X = params[: num_movies * num_features].reshape((num_movies, num_features))
    Theta = params[num_movies * num_features :].reshape((num_users, num_features))

    J = cofi_cost_fn(X, Theta, Y, R, lb)
    grad = cofi_cost_n_grad_fn(X, Theta, Y, R, lb)
    return J, grad


#%% [markdown]
# # Helper

#%%
def check_cost_fn(lb=0):

    X_t = np.random.random((4, 3))
    Theta_t = np.random.random((4, 3))

    Y = X_t @ Theta_t.T
    Y[np.random.random(Y.shape) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    # run gradient checking

    X = np.random.random(X_t.shape)
    Theta = np.random.random(Theta_t.shape)
    num_movies, num_users = Y.shape
    num_features = Theta.shape[1]

    unrolled = np.hstack((X.ravel(), Theta.ravel()))
    numgrad = compute_numerical_gradient(
        lambda t: cofi_cost_fn(t, Y, R, num_users, num_movies, num_features, lb),
        unrolled,
    )
    grad = cofi_grad_fn(unrolled, Y, R, num_users, num_movies, num_features, lb)
    for ngd, gd in zip(numgrad, grad):
        print(f"{ngd:e}\t{gd:e}")

    print(
        f"""The above two columns you get should be very similar.
        (Left-Your Numerical Gradient, Right-Analytical Gradient)
    """
    )

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print(
        f"""If your backpropagation implementation is correct, then 
        the relative difference will be small (less than 1e-9).
        Relative Difference: {diff:e}
    """
    )


#%%
def compute_numerical_gradient(J, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4

    for i in range(theta.size):
        perturb[i] = e
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        numgrad[i] = (loss2 - loss1) / (2 * e)
        perturb[i] = 0
    return numgrad


#%%
def load_movie_list(fname):
    names = []
    with open(fname, encoding="iso-8859-1") as f:
        for line in f.readlines():
            first_spc = line.find(" ")
            names.append(line[first_spc + 1 : -1])
    return names


#%%
def normalize_ratings(Y, R):
    m, n = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros(Y.shape)
    for i in range(m):
        idx = R[i] == 1
        Ymean[i] = Y[i, idx].mean()
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
    return Ynorm, Ymean


#%%
def topk_related_movies(X, mi, k):
    m = X.shape[0]
    xi = X[mi]

    error = ((X - xi) ** 2).sum(1)
    return np.argsort(error)[1 : k + 1]
