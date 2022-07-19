function Z = projectData(X, U, K)
    Z = zeros(size(X, 1), K);

    Z = X * U(:, 1:K); % (m x n) * (n x k)
end