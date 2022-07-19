function [bestEpsilon bestF1] = selectThreshold(yval, pval)

    bestEpsilon = 0;
    bestF1 = 0;
    F1 = 0;

    stepsize = (max(pval) - min(pval)) / 1000;
    for epsilon = min(pval):stepsize:max(pval)
        pred = pval < epsilon;
        tp = sum((pred == 1) & (yval == 1));
        fp = sum((pred == 1) & (yval == 0));
        fn = sum((pred == 0) & (yval == 1));
        prec = tp / (tp + fp);
        rec = tp / (tp + fn);
        F1 = (2 * prec * rec) / (prec + rec);

        if F1 > bestF1
            bestF1 = F1;
            bestEpsilon = epsilon;
        end
    end
end