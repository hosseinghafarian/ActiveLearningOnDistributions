function [u_new]  = u_wmst_from_previous(u_k, pre_nSDP, nSDP, n_S, eind_new, eind_pre)
    % extract structure of the previous matrix
 
    G             = reshape(u_k(1:pre_nSDP*pre_nSDP),pre_nSDP,pre_nSDP);
    p             = u_k(pre_nSDP*pre_nSDP+1:pre_nSDP*pre_nSDP+n_S);
    % change matrix to new form, using cnstData
    
    G_new = map_to_smaller_matrix(G, eind_pre, eind_new);
    % repack to obtain x_new
    u_new         = [reshape(G_new,nSDP*nSDP,1);p];
end