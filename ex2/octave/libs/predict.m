
function p = predict(theta, X)
    m = size(X, 1);
    p = zeros(m, 1);

    p = sigmoid(X * theta);
    for row = 1:m
        if p(row) >= 0.5
            p(row) = 1;
        else
            p(row) = 0;
        end
    end
end