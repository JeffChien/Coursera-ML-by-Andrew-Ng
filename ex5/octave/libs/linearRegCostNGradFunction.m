function [J, grad] = linearRegCostNGradFunction(X, y, theta, lambda)
    J = linearRegCostFn(X, y, theta, lambda);
    grad = linearRegGradFn(X, y, theta, lambda);
end