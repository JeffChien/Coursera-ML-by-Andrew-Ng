function J = computeCost(X, y, theta)
    m = length(y);
    J = 0;
    predictions = X * theta;
    sqrtError = (predictions - y) .^2;
    J = 1/(2*m) * sum(sqrtError);
end
