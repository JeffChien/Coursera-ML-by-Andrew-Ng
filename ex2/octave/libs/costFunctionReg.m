
function [J, grad] = costFunctionReg(theta, X, y, lambda)
    m = length(y);
    J = 0;
    grad = zeros(size(theta));

    [J, grad] = costFunction(theta, X, y);
    v = [0; theta(2:end)];
    J = J + lambda/(2*m) * v'*v;
    grad = grad + lambda/m * v;
end