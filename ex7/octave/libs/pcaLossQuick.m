function loss = pcaLossQuick(S, K)
    n = size(S, 1);
    psum = zeros(n+1, 1);
    for i = 2:n+1
        psum(i) = psum(i-1) + S(i-1, i-1);
    end
    loss = 1 - psum(K+1) / psum(end);
end