function [X_poly] = polyFeatures(X, p)

    m = size(X, 1);
    X_poly = zeros(m, p);

    for i = 1:p
        X_poly(:, i) = X(:, 1) .^ i;
    end
end