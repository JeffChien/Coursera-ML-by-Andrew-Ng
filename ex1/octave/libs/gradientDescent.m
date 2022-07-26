function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
    m = length(y);
    J_history = zeros(num_iters, 1);
    for it = 1:num_iters
       theta = theta - alpha / m * (X' * (X * theta - y));
       J_history(it) = computeCost(X, y, theta);
    end
end
