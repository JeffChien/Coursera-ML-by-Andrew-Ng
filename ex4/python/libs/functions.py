#%%
import math
import os
import sys
import itertools as it
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio

#%%
def patch_c0(X):
    m = X.shape[0]
    return np.c_[np.ones((m, 1)), X]


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
    ax.imshow(display_array.T, cmap=plt.get_cmap("Greys"))
    ax.set_axis_off()
    return ax


#%%
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


#%%
def sigmoid_grad(z):
    g = sigmoid(z)
    return g * (1 - g)


#%%
def restore_Theta(nn_params, size_of_layers):
    st = end = 0
    Thetas = []

    for i in range(len(size_of_layers) - 1):
        cur_sz = size_of_layers[i]
        nxt_sz = size_of_layers[i + 1]
        end = nxt_sz * (cur_sz + 1)
        Theta = nn_params[st : st + end].reshape((nxt_sz, cur_sz + 1))
        Thetas.append(Theta)
        st = st + end
    return Thetas


#%%
def cost_function(nn_params, size_of_layers, X, Y, lb):
    m = Y.shape[0]
    Theta1, Theta2 = restore_Theta(nn_params, size_of_layers)

    # print(f"{Theta1.shape}, {Theta2.shape}")

    # forward porpagation
    a1 = patch_c0(X)

    z2 = a1 @ Theta1.T
    a2 = patch_c0(sigmoid(z2))

    z3 = a2 @ Theta2.T
    a3 = sigmoid(z3)

    J = (Y * np.log(a3) + (1 - Y) * np.log(1 - a3)).sum() * (-1.0 / m)

    # reg terms, no c0
    tmp_Theta1 = Theta1[:, 1:]
    tmp_Theta2 = Theta2[:, 1:]
    reg = (lb / (2 * m)) * ((tmp_Theta1**2).sum() + (tmp_Theta2**2).sum())

    return J + reg


#%%
def grad_function(nn_params, size_of_layers, X, Y, lb):
    m = Y.shape[0]
    Theta1, Theta2 = restore_Theta(nn_params, size_of_layers)

    # forward porpagation
    a1 = patch_c0(X)

    z2 = a1 @ Theta1.T
    a2 = patch_c0(sigmoid(z2))

    z3 = a2 @ Theta2.T
    a3 = sigmoid(z3)

    # Y = m x n
    # a3 = m x n
    # a2 = m x (Sl_2 + 1)
    # a1 = m x (Sl_1 + 1)
    # Theta2 = n x (Sl_2 + 1)
    # Theta1 = Sl_2 x (Sl_1 + 1)

    d3 = a3 - Y  # m x n
    d2 = d3 @ Theta2 * (a2 * (1 - a2))  # m x (Sl_2 + 1)
    d2 = d2[:, 1:]  # m x Sl_2
    D2 = d3.T @ a2  # n x (Sl_2 + 1)
    D1 = d2.T @ a1  # Sl_2 x (Sl_1 + 1)

    temp_Theta1 = np.c_[np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]]
    temp_Theta2 = np.c_[np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]]

    Theta2_grad = D2 / m + (lb / m) * temp_Theta2
    Theta1_grad = D1 / m + (lb / m) * temp_Theta1

    grad = np.hstack((Theta1_grad.ravel(), Theta2_grad.ravel()))
    return grad


#%% [markdown]
# # Random initialization
#
# important for training NN for **symmetry breaking**
#
# one effective strategy is select values for $\Theta^{(l)}$ uniformly in range $[-\epsilon, \epsilon]$
#
# A good choice for $\epsilon$ is
#
# $$
# \epsilon = \frac{\sqrt{6}}{\sqrt{L_{in} + L_{out}}}
# $$

#%%
def rnd_initialize_weights(lin, lout):
    W = np.zeros((lout, lin + 1))
    eps = 0.12
    W = np.random.rand(lout, lin + 1) * 2 * eps - eps
    return W


#%%
def predict(Theta1, Theta2, X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    a1 = patch_c0(X)

    z2 = a1 @ Theta1.T
    a2 = patch_c0(sigmoid(z2))

    z3 = a2 @ Theta2.T
    a3 = sigmoid(z3)

    return np.argmax(a3, 1) + 1


#%% [markdown]
# # Helper

#%%
def compute_numerical_gradient(J, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    eps = 1e-4
    for p in range(theta.size):
        perturb[p] = eps
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        numgrad[p] = (loss2 - loss1) / (2 * eps)
        perturb[p] = 0
    return numgrad


#%%
def debug_initialize_weights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))
    W = np.sin(np.arange(0, W.size)).reshape(W.shape) / 10.0
    return W


#%%
def checkNNGradients(lb):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    size_of_layers = [input_layer_size, hidden_layer_size, num_labels]
    m = 5

    Theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    Theta2 = debug_initialize_weights(num_labels, hidden_layer_size)

    X = debug_initialize_weights(m, input_layer_size - 1)
    y = 1 + np.arange(0, m) % num_labels
    Y = np.zeros((m, num_labels))
    for i in range(m):
        Y[i, y[i] - 1] = 1

    nn_params = np.hstack((Theta1.ravel(), Theta2.ravel()))
    costfn = lambda p: cost_function(p, size_of_layers, X, Y, lb)

    grad = grad_function(nn_params, size_of_layers, X, Y, lb)

    numgrad = compute_numerical_gradient(costfn, nn_params)

    for ng, gg in zip(numgrad, grad):
        print(f"{ng:f}\t{gg:f}")

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print(
        f"""
        If your backpropagation implementation is correct, then
        the relative difference will be small (less then 1e-9).
        Relative Difference: {diff}
    """
    )
