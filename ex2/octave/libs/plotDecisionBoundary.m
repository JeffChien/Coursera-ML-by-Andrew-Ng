
function plotDecisionBoundary(theta, X, y)
    plotData(X(:, 2:3), y);
    hold on;
    plot_x = [min(X(:, 2))-2, max(X(:, 2)) + 2];
    plot_y = (theta(2) .* plot_x + theta(1)) .* (-1 / theta(3));
    plot(plot_x, plot_y);
    hold off;
end