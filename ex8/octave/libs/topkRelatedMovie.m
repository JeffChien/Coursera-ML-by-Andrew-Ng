function ix = topkRelatedMovie(X, mi, k)
    m = size(X, 1);
    xi = X(mi, :);

    error = sum((X - xi) .^2, 2);
    [r, ix] = sort(error);
    ix = ix(2:k+1);
end