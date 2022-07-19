import os
import sys
import itertools as it
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.optimize import fmin_cg
from scipy.optimize import fmin_bfgs

#%%
def linear_reg_cost_fn(theta, X, y, lb):
    m = X.shape[0]
    J = 0

    error = theta @ X.T - y
    J = error @ error.T / (2 * m)

    v = theta.copy()
    v[0] = 0

    reg = (lb / (2 * m)) * (v @ v.T)
    J += reg
    return J


#%%
def linear_reg_grad_fn(theta, X, y, lb):
    m = X.shape[0]
    grad = np.zeros(theta.shape)
    v = theta.copy()
    v[0] = 0

    grad = ((theta @ X.T - y) @ X) / m
    reg = (lb / m) * v
    grad += reg
    return grad


#%%
def train_linear_reg(X, y, lb):
    m, n = X.shape
    initial_theta = np.zeros(n)
    theta = fmin_cg(
        # theta = fmin_bfgs(
        linear_reg_cost_fn,
        initial_theta,
        linear_reg_grad_fn,
        args=(X, y, lb),
        maxiter=200,
    )
    return theta


#%%
def learning_curve(X, y, Xval, yval, lb):
    m, n = X.shape

    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(m):
        XX = X[: i + 1]
        yy = y[: i + 1]
        theta = train_linear_reg(XX, yy, lb)
        error_train[i] = linear_reg_cost_fn(theta, XX, yy, 0)
        error_val[i] = linear_reg_cost_fn(theta, Xval, yval, 0)
    return error_train, error_val


#%%
def rnd_learning_curve(X, y, Xval, yval, lb):
    m, n = X.shape

    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(m):
        error_sample_train = np.zeros(i + 1)
        error_sample_val = np.zeros(i + 1)
        for j in range(10):
            train_seq = np.random.choice(range(m), i + 1)
            XX = X[train_seq]
            yy = y[train_seq]
            theta = train_linear_reg(XX, yy, lb)
            error_sample_train[i] = linear_reg_cost_fn(theta, XX, yy, 0)
            error_sample_val[i] = linear_reg_cost_fn(theta, Xval, yval, 0)
        error_train[i] = error_sample_train.mean()
        error_val[i] = error_sample_val.mean()
    return error_train, error_val


#%%
def poly_features(X, p):
    m, n = X.shape
    X_poly = np.zeros((m, p))

    for i in range(p):
        X_poly[:, i] = X[:, 0] ** (i + 1)
    return X_poly


#%%
def feature_normalize(X):
    mu = X.mean(0)
    X_norm = X - mu

    sigma = X_norm.std(0)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma


#%%
def poly_fit(ax, min_x, max_x, mu, sigma, theta, p):
    x = np.linspace(min_x - 15, max_x + 25, 100)
    X_poly = poly_features(x[np.newaxis, :].T, p)
    X_poly = (X_poly - mu) / sigma
    X_poly = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]

    ax.plot(x, theta @ X_poly.T, "--", linewidth=2)
    return ax


#%%
def validation_curve(X, y, Xval, yval):
    lb_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    error_train = np.zeros(len(lb_vec))
    error_val = np.zeros(len(lb_vec))

    for i, lb in enumerate(lb_vec):
        theta = train_linear_reg(X, y, lb)

        print(f"{lb}, {theta.shape}, {Xval.T.shape}, {yval.shape}")

        error_train[i] = linear_reg_cost_fn(theta, X, y, 0)
        error_val[i] = linear_reg_cost_fn(theta, Xval, yval, 0)
    return lb_vec, error_train, error_val
