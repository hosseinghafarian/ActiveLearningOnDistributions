function [Rreginv_cell_all] = compute_R_reginv_forall(nA, R, F_A, uF_A, idx_dF_A, midx_A, thau)
Rreginv_cell_all = cell(midx_A, 1);

for i=1: midx_A
    [stidx_A, enidx_A, nX_i, map_i] = get_dist_info(idx_dF_A, uF_A, midx_A, i);
    Rreginv_cell_all{i} = getR_reginv_dist( nX_i, stidx_A, enidx_A, thau);
end
    function R_reginv = getR_reginv_dist(nX_i, stidx_A, enidx_A, thau)
        R_reginv = inv(R(stidx_A:enidx_A-1, stidx_A:enidx_A-1)+ thau/nX_i*eye(nX_i));
    end
end