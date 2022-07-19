function [J, grad] = cofiCostnGradFn(params, Y, R, num_users, num_movies, num_features, lambda)
    X = reshape(params(1:num_movies*num_features), num_movies, num_features);
    Theta = reshape(params(num_movies*num_features+1:end), num_users, num_features);

    J = 0;
    X_grad = zeros(size(X));
    Theta_grad = zeros(size(Theta));

    J = cofiCostFn(X, Theta, Y, R, lambda);
    grad = cofiGradFn(X, Theta, Y, R, lambda);
end