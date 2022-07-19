function [theta] = trainLinearReg(X, y, lambda)
    initial_theta = zeros(size(X, 2), 1);
    costFn = @(t) linearRegCostNGradFunction(X, y, t, lambda);
    options = optimset('MaxIter', 200, 'GradObj', 'on');
    theta = fmincg(costFn, initial_theta, options);
    % theta = fminunc(costFn, initial_theta, options);
end