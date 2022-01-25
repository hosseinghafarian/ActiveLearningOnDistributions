function [R, b, size_of_b]                   = eq_AEC_diag_of_extended(c_pwo,elements_of_diag)
    % cConstraint=[cConstraint, diag(G_plus(setQuery,setQuery))==q];
    % Constraint: diag(G_qq)==q
global cnstData
    t     = 1;
    n_e   = numel(cnstData.extendInd);
    R     = spalloc(cnstData.nConic,n_e,2*n_e);
    b     = zeros(n_e,1);
    for k = cnstData.extendInd
       R1        = sparse([k,k,cnstData.nSDP],[k,cnstData.nSDP,k],[1,-0.5,-0.5],cnstData.nSDP,cnstData.nSDP);
       R(:,t)    = [reshape(R1,cnstData.nSDP*cnstData.nSDP,1)',zeros(cnstData.n_S,1)'];
       b(t,1)    = 0;
       t = t+1; 
    end 
    size_of_b    = numel(cnstData.extendInd);
end