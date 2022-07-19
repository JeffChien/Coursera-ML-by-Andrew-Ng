#%%
import numpy as np
import numpy.typing as npt
import matplotlib as mpl
import matplotlib.pyplot as plt


#%%
def plot_data(X: npt.NDArray, y: npt.NDArray):
    pos = y == 1
    neg = y == 0
    fig, ax = plt.subplots()
    ax.plot(X[pos, 0], X[pos, 1], "k+", linewidth=2, markersize=7)
    ax.plot(X[neg, 0], X[neg, 1], "ko", markerfacecolor="y", markersize=7)
    return ax


#%%[markdown]
# # Decision boundary, line
#
# recall the hypothesis of logistic regression
# $$
# h(z) = \frac{1}{1+e^{-z}}
# $$
# and
# $$
# z = \theta^Tx
# $$
#
# as definition, we separate the 2 classes when $h(z) = \frac{1}{2}$, which is also when $z=0$
#
# now, we have 2 features $z = \theta_0 + \theta_1x_1 + \theta_2x_2$
#
# $\theta_0 + \theta_1x_1 + \theta_2x_2 = 0$
#
# $x_2' = -\frac{\theta_1x_1 + \theta_0}{\theta_2}$
#
# $x_1' = [\min(x_1)-2, \max(x_1)+2]$

#%%
def plot_decision_boundary(theta: npt.NDArray, X: npt.NDArray, y: npt.NDArray):
    ax = plot_data(X[:, 1:], y)
    xx = np.array([min(X[:, 1]) - 2, max(X[:, 1]) + 2])
    yy = (theta[1] * xx + theta[0]) * (-1.0 / theta[2])
    ax.plot(xx, yy)
    return ax


def plot_decision_boundary2(theta: npt.NDArray, X: npt.NDArray, y: npt.NDArray):
    ax = plot_data(X[:, 1:], y)
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((u.size, v.size))

    for i in range(u.size):
        for j in range(v.size):
            z[i, j] = map_features(u.take([i]), v.take([j])) @ theta.T
    z = z.T
    uu, vv = np.meshgrid(u, v)
    ax.contour(uu, vv, z, 0)
    return ax


# %% [markdown]
# # Sigmoid function
#
# $$
# h_\theta(x) = g(\theta^Tx)
# $$
# $$
# g(z) = \frac{1}{1 + e^{-z}}
# $$
# %%
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# %% [markdown]
# # Cost function and Gradiant
# $$
# J(\theta) = -\frac{1}{m}\left[\sum^m_{i=1} y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))\right]
# $$
#
# $$
# \frac{\partial}{\partial\theta_j}J(\theta) = \frac{1}{m}\sum^m_{i=1}(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}
# $$

# %%
def cost_function(theta, X, y):
    """
    row vec: theta, y
    """
    m = y.size
    g = sigmoid(X.dot(theta.T))  # col vec
    J = (y.dot(np.log(g)) + (1 - y).dot(np.log(1 - g))) * -1 / m
    return J


def cost_function_reg(theta, X, y, la):
    """
    row vec: theta, y
    """
    m = X.shape[0]
    J = cost_function(theta, X, y)
    v = theta.copy()
    v[0] = 0
    return J + la / (2 * m) * (v @ v.T)


def gradiant_function(theta, X, y):
    """
    row vec: theta, y
    """
    m = y.size
    g = sigmoid(X.dot(theta.T))  # col vec
    grad = (g.T - y).dot(X) / m
    return grad


def gradiant_function_reg(theta, X, y, la):
    """
    row vec: theta, y
    """
    m = y.size
    v = theta.copy()
    v[0] = 0
    grad = gradiant_function(theta, X, y)
    return grad + la / m * v


#%%
def predict(theta, X):
    m = X.shape[0]
    p = np.zeros(m)
    p: npt.NDArray = sigmoid(X @ theta.T)
    for i in range(m):
        p[i] = float(p[i] >= 0.5)
    return p


#%%
def map_features(X1, X2):
    degree = 6
    m = X1.shape[0]
    out = np.ones((m, 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            v = (X1 ** (i - j)) * (X2**j)
            out = np.c_[(out, v)]
    return out
