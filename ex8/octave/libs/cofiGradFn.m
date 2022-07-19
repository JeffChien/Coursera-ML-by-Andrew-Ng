function grad = cofiGradFn(X, Theta, Y, R, lambda)
    % X = m x n
    % Theta = u x n
    % Y = m x u
    % R = m x u

    X_grad = zeros(size(X));
    Theta_grad = zeros(size(Theta));

    common = (X * Theta' - Y) .* R;

    X_grad = common * Theta + lambda * X;
    Theta_grad = common' * X + lambda * Theta;

    grad = [X_grad(:); Theta_grad(:)];
end