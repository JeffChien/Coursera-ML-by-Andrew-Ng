function plotFeatureHist(X, ncols=3, nbins=25)
    nfeatures = size(X, 2);
    ncols = min(nfeatures, ncols);
    nrows = ceil(nfeatures / ncols);
    for i = 1:nfeatures
        subplot(nrows, ncols, i);
        hist(X(:, i), nbins);
        title(sprintf('feature x%d', i));
    end
end