function p = predict(Theta1, Theta2, X)
    m = size(X, 1);
    num_labels = size(Theta2, 1);
    p = zeros(m, 1);

    X = [ones(m, 1) X];

    a1 = X;

    z2 = a1 * Theta1';
    a2 = [ones(m, 1) sigmoid(z2)];

    z3 = a2 * Theta2';
    a3 = sigmoid(z3);

    [value, index] = max(a3, [], 2);
    p = index;
end