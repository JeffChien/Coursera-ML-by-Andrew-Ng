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
def find_closest_centroinds(X, centroids):
    m = X.shape[0]
    K = centroids.shape[0]
    idx = np.zeros(m)

    for i in range(m):
        x = X[i]
        mse = ((x - centroids) ** 2).sum(1)
        idx[i] = np.argmin(mse)
    return idx


#%%
def compute_centroids(X, idx, K):
    m, n = X.shape

    # centroids is (k x n)
    centroids = np.zeros((K, n))

    for i in range(K):
        ck = idx == i
        centroids[i, :] = X[ck, :].sum(0) / ck.sum()
    return centroids


#%%
def kmean_init_centroids(X, K):
    randidx = np.random.choice(X.shape[0], K)
    return X[randidx, :]


#%%
def feature_normalize(X):
    mu = X.mean(0)
    X_norm = X - mu
    sigma = X_norm.std(0)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma


#%%
def pca(X):
    m, n = X.shape

    Sigma = (X.T @ X) / m
    U, S, V = np.linalg.svd(Sigma)
    return U, S


#%%
def project_data(X, U, K):
    Z = X @ U[:, :K]
    return Z


#%%
def recover_data(Z, U, K):
    X_rec = Z @ U[:, :K].T
    return X_rec


#%%
def pca_loss(X, Z, U, K):
    m = X.shape[0]
    X_rec = recover_data(Z, U, K)
    loss = ((X_rec - X) ** 2).sum() / (X**2).sum()
    return loss


#%%
def pca_loss_quick(S, K):
    n = S.shape[0]
    psum = [0] * (n + 1)
    for i in range(1, n + 1):
        psum[i] = psum[i - 1] + S[i - 1]
    loss = 1.0 - psum[K] / psum[-1]
    return loss


def pca_k_in_maxloss(S, max_loss=0.01):
    n = S.shape[0]
    lo, hi = 1, n + 1
    psum = [0] * (n + 1)
    for i in range(1, n + 1):
        psum[i] = psum[i - 1] + S[i - 1]

    while lo < hi:
        mid = lo + (hi - lo) // 2
        loss = 1.0 - psum[mid] / psum[-1]
        if loss <= max_loss:
            hi = mid
        else:
            lo = mid + 1
    return lo


#%% [markdown]
# # Helper

#%%
def run_kmeans(X, initial_centroids, max_iters, plot_progress=False):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros(m)

    for i in range(max_iters):
        idx = find_closest_centroinds(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx


#%%
def plot_data_poits(X, idx, K):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=idx, cmap=plt.cm.Paired)
    return ax


def draw_line(ax, p1, p2, *args, **kwargs):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)
    return ax


# %%
def display_data(X, example_width=None):
    if example_width is None:
        example_width = int(round(math.sqrt(X.shape[1])))

    m, n = X.shape

    example_height = n // example_width
    display_rows = int(math.floor(math.sqrt(m)))
    display_cols = int(math.ceil(m / display_rows))

    pad = 1
    display_array = -np.ones(
        (
            pad + display_rows * (example_height + pad),
            pad + display_cols * (example_width + pad),
        )
    )

    for k, (i, j) in enumerate(it.product(range(display_rows), range(display_cols))):
        if k == m:
            break

        max_val = max(abs(X[k, :]))
        ri = pad + i * (example_height + pad)
        cj = pad + j * (example_width + pad)
        display_array[ri : ri + example_height, cj : cj + example_width] = (
            X[k, :].reshape(example_height, example_width) / max_val
        )
    fig, ax = plt.subplots()
    ax.imshow(display_array.T, cmap=plt.get_cmap("gray"))
    ax.set_axis_off()
    return ax
