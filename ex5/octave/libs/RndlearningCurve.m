function [error_train, error_val] = RndlearningCurve(X, y, Xval, yval, lambda)
    m = size(X, 1);
    error_train = zeros(m, 1);
    error_val = zeros(m, 1);

    for i = 1:m
        error_sample_train = zeros(i, 1);
        error_sample_val = zeros(i, 1);
        for j=1:m/i
            train_seq = randperm(m, i);
            theta = trainLinearReg(X(train_seq, :), y(train_seq), lambda);
            error_sample_train(i) = linearRegCostFn(X(train_seq, :), y(train_seq), theta, 0);
            error_sample_val(i) = linearRegCostFn(Xval, yval, theta, 0);
        end
        error_train(i) = mean(error_sample_train);
        error_val(i) = mean(error_sample_val);
    end
end