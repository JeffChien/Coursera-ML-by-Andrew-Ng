function [U, S] = pca(X)
    [m, n] = size(X);

    U = zeros(n);
    S = zeros(n);

    Sigma = (X' * X) ./ m;
    [U, S, V] = svd(Sigma);
end