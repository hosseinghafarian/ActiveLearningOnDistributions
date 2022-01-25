function [R, KB, Rreginv_cells, mu_R_cells] = compute_KBayes_m( A, F_A, gamma, gamma_r, thau)
nA = numel(F_A);
R = R_forall(nA, A, gamma_r);
[midx_A, idx_dF_A, uF_A] = compute_dist_idx_m(nA, F_A);
KB = zeros(midx_A, midx_A);
[mu_R_cells, Rreginv_cells ] = comp_mu_R_all(nA, A, R, F_A, uF_A, idx_dF_A, midx_A, thau, gamma);
for i=1:midx_A
    [stidx_A_i, enidx_A_i, nX_i, map_i] = get_dist_info(idx_dF_A, uF_A, midx_A, i);
    for j=1:i
         [stidx_A_j, enidx_A_j, nX_j, map_j] = get_dist_info(idx_dF_A, uF_A, midx_A, j);
         KB(i,j) = comp_KB_ij(stidx_A_i, enidx_A_i, stidx_A_j, enidx_A_j);
         KB(j,i) = KB(i,j);
    end
end
function val = comp_KB_ij(stidx_A_i, enidx_A_i, stidx_A_j, enidx_A_j)
   val = mu_R_cells{i}'*R(stidx_A_i:enidx_A_i-1, stidx_A_j:enidx_A_j-1)*mu_R_cells{j};
end
end