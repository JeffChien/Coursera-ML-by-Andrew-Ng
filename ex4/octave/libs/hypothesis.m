function HL = hypothesis(Theta1, Theta2, X)
    m = size(X, 1);
    a1 = [ones(m, 1) X];

    z2 = a1 * Theta1';
    a2 = [ones(m, 1) sigmoid(z2)];

    z3 = a2 * Theta2';
    a3 = sigmoid(z3);

    HL = [a2(:); a3(:)];
end