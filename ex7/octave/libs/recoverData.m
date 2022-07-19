function X_rec = recoverData(Z, U, K)

    X_rec = zeros(size(Z, 1), size(U, 1)); # m x n

    # Ureduced = (n x k)
    # Z = (m x k)

    X_rec = Z * U(:, 1:K)';

end