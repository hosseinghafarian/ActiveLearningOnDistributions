function [mbe_cell_all, FA_map, FA_n] = meanembed_fval_dist(nA, A, F_A, uF_A, idx_dF_A, midx_A, gamma)
mbe_cell_all = cell(midx_A, 1);
FA_map       = zeros(midx_A,1);
FA_n         = zeros(midx_A,1);
for i=1: midx_A
    [stidx_A, enidx_A, nX_i, map_i] = get_dist_info(idx_dF_A, uF_A, midx_A, i);
    FA_map(i) = map_i;
    FA_n(i) = nX_i;
    mbe_cell_all{i} = meanembed_fval_at_all(gamma, nX_i, A(:, stidx_A:enidx_A-1));
end

end