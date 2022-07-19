function [J grad] = nnCostFunction(nn_params, ...
                        input_layer_size, ...
                        hidden_layer_size, ...
                        num_labels, ...
                        X, y, lambda)

    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                    hidden_layer_size, (input_layer_size + 1));
    Theta2 = reshape(nn_params(1 + hidden_layer_size * (input_layer_size + 1): end), ...
                    num_labels, (hidden_layer_size + 1));

    m = size(X, 1);

    Y = zeros(m, num_labels);
    for i=1:m
        Y(i, y(i, 1)) = 1;
    end

    # forward propagation
    a1 = [ones(m, 1) X];

    z2 = a1 * Theta1';
    a2 = [ones(m, 1) sigmoid(z2)];

    z3 = a2 * Theta2';
    a3 = sigmoid(z3);

    J = costFunction(Theta1, Theta2, a3, Y, lambda);

    grad = nnGradFunction(Theta1, Theta2, a1, a2, a3, Y, lambda);

end