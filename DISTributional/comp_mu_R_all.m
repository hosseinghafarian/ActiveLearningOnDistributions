function [mu_R_i_cells, Rreginv_cell_all ] = comp_mu_R_all(nA, A, R, F_A, uF_A, idx_dF_A, midx_A, thau, gamma)
[Rreginv_cell_all] = compute_R_reginv_forall(nA, R, F_A, uF_A, idx_dF_A, midx_A, thau);

[mbe_cell_all, ~, ~] = meanembed_fval_dist(nA, A, F_A, uF_A, idx_dF_A, midx_A, gamma);
mu_R_i_cells = cell(midx_A, 1);
for i=1:midx_A
    mu_R_i_cells{i} = Rreginv_cell_all{i}*mbe_cell_all{i};
end

end