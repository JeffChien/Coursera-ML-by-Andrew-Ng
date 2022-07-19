function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
    m = length(y);
    J_history = zeros(num_iters, 1);

    for it = 1:num_iters
        Delta = ((X * theta - y)' * X)';
        theta = theta - (alpha / m * Delta);
        J_history(it) = computeCostMulti(X, y, theta);
    end
end
