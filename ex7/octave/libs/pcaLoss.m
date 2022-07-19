function loss = pcaLoss(X, Z, U, K)
    m = size(X, 1);
    X_rec = recoverData(Z, U, K);

    loss = sum(sum((X_rec - X) .^ 2)) / sum(sum(X .^ 2));
end