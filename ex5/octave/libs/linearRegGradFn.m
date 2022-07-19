function grad = linearRegGradFn(X, y, theta, lambda)
    m = length(y);
    grad = zeros(size(theta));

    v = theta;
    v(1) = 0;

    grad = ((X * theta - y)' * X)' / m;
    reg = (lambda / m) * v;
    grad = grad + reg;
end