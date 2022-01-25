function [midx_A, idx_dF_A, uF_A] = compute_dist_idx_m(nA, F_A)

nzidx_dF_A = diff([0, F_A]);
logic    = nzidx_dF_A~=0;
tmpidx = find(nzidx_dF_A);
midx_A = numel(tmpidx);
uF_A   = F_A(logic);
idx_dF_A = [find(nzidx_dF_A), nA+1];
end