function grad = nnGradFunction(Theta1, Theta2, a1, a2, a3, Y, lambda)
    m = size(Y, 1);

    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));

    D2 = zeros(size(Theta2_grad));
    D1 = zeros(size(Theta1_grad));

    grad = 0;

    % y = 5000 x 10
    % a3 = 5000 x 10
    % a2 = 5000 x 26
    % a1 = 5000 x 401
    % Theta2 = 10 x 26
    % Theta1 = 25 x 401

    d3 = a3 - Y; % 5000 x 10
    d2 = d3 * Theta2 .* (a2 .* (1 - a2));
    d2 = d2(:, 2:end); % 5000 x 25
    D2 = d3' * a2; % 10 x 26
    D1 = d2' * a1; % 25 x 401

    tempTheta1 = [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
    tempTheta2 = [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

    Theta2_grad = D2./m + (lambda/m) * tempTheta2;
    Theta1_grad = D1./m + (lambda/m) * tempTheta1;

    grad = [Theta1_grad(:); Theta2_grad(:)];
end