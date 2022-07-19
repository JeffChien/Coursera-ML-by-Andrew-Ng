function centroids = computeCentroids(X, idx, K)
    [m, n] = size(X);

    centroids = zeros(K, n);

    for i = 1:K
        ck = find(idx == i);
        centroids(i, :) = sum(X(ck, :)) / numel(ck);
    end
end