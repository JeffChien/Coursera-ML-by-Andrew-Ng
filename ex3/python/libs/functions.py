#%%
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools as it
from scipy.optimize import fmin_cg
from scipy.optimize import fmin_bfgs


#%%
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
    ax.imshow(display_array.T, cmap=plt.get_cmap("Greys"))
    ax.set_axis_off()
    return ax


#%%
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


#%%
def cost_function(theta, X, y, lb):
    m = X.shape[0]
    g = sigmoid(X @ theta.T)
    J_noreg = (-1.0 / m) * (y @ np.log(g) + (1 - y) @ np.log(1 - g))
    v = theta.copy()
    v[0] = 0
    reg_fix = (lb / (2 * m)) * (v @ v.T)
    return J_noreg + reg_fix


def grad_function(theta, X, y, lb):
    m = X.shape[0]
    g = sigmoid(X @ theta.T).T

    v = theta.copy()
    v[0] = 0
    reg_fix = (lb / m) * v

    grad_noreg = ((g - y) @ X) / m
    return grad_noreg + reg_fix


#%%
def one_vs_all(X, y, num_labels, lb):
    m, n = X.shape

    all_theta = np.zeros((num_labels, n + 1))
    X = np.c_[(np.ones((m, 1)), X)]

    initial_theta = np.zeros(n + 1)
    for c in range(num_labels):
        theta = fmin_cg(
            cost_function,
            initial_theta,
            grad_function,
            args=(X, y == c + 1, lb),
            maxiter=50,
        )
        all_theta[c, :] = theta
    return all_theta


#%%
def predict_one_vs_all(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    X = np.c_[(np.ones((m, 1)), X)]

    temp = X @ all_theta.T
    return np.argmax(temp, axis=1) + 1


#%%
def predict(Theta1, Theta2, X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    X = np.c_[(np.ones((m, 1)), X)]

    a1 = X

    z2 = a1 @ Theta1.T
    a2 = np.c_[(np.ones((m, 1)), sigmoid(z2))]

    z3 = a2 @ Theta2.T
    a3 = sigmoid(z3)

    return np.argmax(a3, axis=1) + 1
