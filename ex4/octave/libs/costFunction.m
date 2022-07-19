function J = costFunction(Theta1, Theta2, g, Y, lambda)
    m = size(Y, 1);

    J = sum(sum(Y .* log(g) + (1-Y).*log(1-g)))*(-1/m);
    tempTheta1 = Theta1(:, 2:end);
    tempTheta2 = Theta2(:, 2:end);
    reg = (lambda / (2*m)) * sum([sum(sum(tempTheta1 .^ 2)) sum(sum(tempTheta2 .^2))]);
    J = J + reg;
end