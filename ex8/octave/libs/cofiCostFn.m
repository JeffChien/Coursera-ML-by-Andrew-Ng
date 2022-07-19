function J = cofiCostFn(X, Theta, Y, R, lambda)
    % X = m x n
    % Theta = u x n
    % Y = m x u
    % R = m x u
    J = 0.5 * sum(sum(((X * Theta' - Y) .^ 2) .* R));

    % reg
    reg = 0.5 * lambda * (sum(sum(Theta .^2)) + sum(sum(X .^2)));

    J += reg;
end