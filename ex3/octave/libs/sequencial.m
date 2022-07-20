function A_out = sequencial(A, Ws, g)
    % Vectorized NN forward propagation
    %
    % A is input matrix
    % Ws is a cell array store transposed Theta
    % g is activation function

    for i = 1:length(Ws)
        A = dense(A, Ws{i}, g);
    end
    A_out = A;
end
