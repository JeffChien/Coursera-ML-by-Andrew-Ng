function [mu, sigma2] = estimateGaussian(X)
    [m, n] = size(X);

    mu = zeros(n, 1);
    sigma2 = zeros(n, 1);

    mu = (sum(X) ./ m)';
    sigma2 = (sum((X - mu') .^ 2) ./ m)';
end