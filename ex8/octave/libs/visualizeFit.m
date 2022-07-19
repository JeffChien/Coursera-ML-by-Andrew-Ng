function visualizeFit(X, mu, sigma2, density_fn)

    [X1, X2] = meshgrid(0:.5:35);

    Z = density_fn([X1(:) X2(:)], mu, sigma2);
    Z = reshape(Z, size(X1));

    plot(X(:, 1), X(:, 2), 'bx');
    hold on;
    if (sum(isinf(Z)) == 0)
        contour(X1, X2, Z, 10 .^(-20:3:0)');
    end
    hold off;
end