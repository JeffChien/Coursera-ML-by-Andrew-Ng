function A_out = dense(A, W, g)
    % Vectorized NN layer
    %
    % A is input matrix
    % W is transposed Theta
    % g is activation function

    m = size(A, 1);
    Z = [ones(m, 1) A] * W;
    A_out = g(Z);
end
