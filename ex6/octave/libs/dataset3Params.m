function [C, sigma] = dataset3Params(X, y, Xval, yval)

    C = 1;
    sigma = 0.3;

    candidates = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
    Cs = candidates;
    sigmas = candidates;
    error_val = zeros(length(Cs), length(sigmas));

    for i = 1:length(candidates)
        for j = 1:length(candidates)
            model = svmTrain(X, y, Cs(i), @(x1, x2) gaussianKernel(x1, x2, sigmas(j)));
            pred = svmPredict(model, Xval);
            error_val(i, j) = mean(double(pred ~= yval));
        end
    end

    [value, ind] = min(error_val(:));
    [i, j] = ind2sub(size(error_val), ind);
    C = Cs(i);
    sigma = sigmas(j);
end