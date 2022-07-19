function idx = findClosestCentroids(X, centroids)

    m = size(X, 1);
    K = size(centroids, 1);
    idx = zeros(size(X, 1), 1);

    for i = 1:m
        x = X(i, :);
        squared_error = sum((x - centroids) .^ 2, 2);
        [val, muk] = min(squared_error);
        idx(i) = muk;
    end
end