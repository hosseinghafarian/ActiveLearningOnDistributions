function [stidx_A, enidx_A, nX_i, map_i] = get_dist_info(idx_dF_A, uF_A, midx_A, i)
stidx_A = idx_dF_A(i);
enidx_A = idx_dF_A(i+1);
nX_i = enidx_A-stidx_A;
map_i = uF_A(i);
end