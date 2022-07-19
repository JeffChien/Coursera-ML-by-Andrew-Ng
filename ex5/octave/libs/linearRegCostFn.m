function J = linearRegCostFn(X, y, theta, lambda)
    m = length(y);
    J = 0;

    error = X * theta - y;
    J = (error' * error) / (2*m);
    v = theta;
    v(1) = 0;

    reg = (lambda/(2*m)) * (v' * v);
    J = J + reg;
end