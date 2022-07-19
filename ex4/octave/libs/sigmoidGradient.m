function gd = sigmoidGradient(z)
    gd = sigmoid(z) .* (1 - sigmoid(z));
end