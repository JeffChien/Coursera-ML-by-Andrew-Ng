function [J, grad] = lrCostFunction(theta, X, y, lambda)
    m = length(y);
    J = 0;
    grad = zeros(size(theta));

    g = sigmoid(X * theta);

    J_noreg = (-1.0/m) * (y' * log(g) + (1-y)' * log(1-g));

    % reg term
    v = theta;
    v(1) = 0;

    reg_fix = (lambda / (2*m)) * (v' * v);

    J = J_noreg + reg_fix;

    grad = ((g-y)' * X)'/m + (lambda/m) * v;
end