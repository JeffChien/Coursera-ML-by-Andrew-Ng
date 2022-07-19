function J = computeCostMulti(X, y, theta)
    m = length(y);
    Delta = (X * theta - y);
    sqrtError = Delta' * Delta;
    J = 1/(2*m) * sqrtError;
end
