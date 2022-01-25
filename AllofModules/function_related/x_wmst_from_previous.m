function [x_new]  = x_wmst_from_previous(x_k, x_st_IC, x_st_IV)
global cnstData
    % extract structure of the previous matrix
    pre_nSDP      = cnstData.pre_nSDP;
    nSDP          = cnstData.nSDP;
    n_S           = cnstData.n_S; 
    w_obeta       = x_k.w_obeta;
    st            = [x_st_IC;x_st_IV];
    G             = reshape(x_k.u(1:pre_nSDP*pre_nSDP),pre_nSDP,pre_nSDP);
    p             = x_k.u(pre_nSDP*pre_nSDP+1:pre_nSDP*pre_nSDP+n_S);
    % change matrix to new form, using cnstData
    [eind_new, eind_pre] = get_map_extndind(cnstData.initL);
    G_new = map_to_smaller_matrix(G, eind_pre, eind_new);
    % repack to obtain x_new
    x_new.u       = [reshape(G_new,nSDP*nSDP,1);p];
    x_new.w_obeta = w_obeta;
    x_new.st      = st;
end