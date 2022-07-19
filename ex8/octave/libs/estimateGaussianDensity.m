function p = estimateGaussianDensity(X, mu, sigma2)
    n = length(mu);
    X = X - mu';
    p = prod((2*pi)^(-0.5)*(sigma2').^(-0.5) .* exp(-0.5*(X .* X)./sigma2'), 2);
end