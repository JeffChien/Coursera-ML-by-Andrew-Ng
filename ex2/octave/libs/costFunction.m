function [J, grad] = costFunction(theta, X, y)
    m = length(y);
    J = 0;
    grad = zeros(size(theta));
    g = sigmoid(X * theta);
    J = (y' * log(g) + (1-y)'*log(1-g)) * -1 / m;
    grad = ((g - y)' * X)' / m;
end