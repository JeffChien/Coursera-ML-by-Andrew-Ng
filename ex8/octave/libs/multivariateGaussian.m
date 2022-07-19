function p = multivariateGaussian(X, mu, sigma2)
    n = length(mu);
    if(size(sigma2, 2) == 1 || size(sigma, 1) == 1)
        Sigma2 = diag(sigma2);
    end

    X = X - mu';
    p = (2 * pi)^(-n/2) * det(Sigma2)^(-0.5) * exp(-0.5*sum(X*pinv(Sigma2) .* X, 2));
end