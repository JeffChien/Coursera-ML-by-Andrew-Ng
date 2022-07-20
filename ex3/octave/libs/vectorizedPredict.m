function p = vectorizedPredict(A, Ws, g)
    % Vectorized NN predict function
    %
    % A is input matrix
    % Ws is a cell array store transposed Theta
    % g is activation function

    f_x = sequencial(A, Ws, g);
    [value, p] = max(f_x, [], 2);
end
