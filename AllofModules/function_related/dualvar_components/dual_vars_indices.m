function [ind_EC, ind_EV, ind_IC, ind_IV, ind_S, ind_Z, ind_v] = dual_vars_indices(operators)
global cnstData
    n_Cone = cnstData.nConic;
    st     = 0;
    ind_EC = st+1:st+operators.n_AEC;
    st     = st+ operators.n_AEC;
    ind_EV = st+1:st+ operators.n_AEV;
    st     = st+ operators.n_AEV;
    ind_IC = st+1:st+operators.n_AIC;
    st     = st+operators.n_AIC;
    ind_IV = st+1:st+operators.n_AIV;
    st     = st+operators.n_AIV;
    ind_S  = st+1:st+n_Cone;
    st     = st+n_Cone;
    ind_Z  = st+1:st+n_Cone;
    st     = st+n_Cone;
    ind_v  = st+1:st+operators.n_AIC+operators.n_AIV;
end