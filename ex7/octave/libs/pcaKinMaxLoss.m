function K = pcaKinMaxLoss(S, max_loss=0.01)
    n = size(S, 1);
    lo = 1;
    hi = n + 1;

    psum = zeros(n+1, 1);
    for i = 2:n+1
        psum(i) = psum(i-1) + S(i-1, i-1);
    end

    while (lo < hi)
        mid = floor(lo + (hi - lo) / 2);
        loss = 1 - psum(mid+1) / psum(end);
        if loss <= max_loss
            hi = mid;
        else
            lo = mid + 1;
        end
    endwhile
    K = lo;
end