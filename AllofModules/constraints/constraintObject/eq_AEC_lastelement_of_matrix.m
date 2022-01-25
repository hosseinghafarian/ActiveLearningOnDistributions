function [R, b]                              = eq_AEC_lastelement_of_matrix(c_mul_pAndw_o,elements_of_diag)  
    % cConstraint= [cConstraint,G_plus(nap+1,nap+1)==1];
    % Constraint: G(nSDP,nSDP) = 1
global cnstData
dummy_pag = zeros(cnstData.n_S,1);
    nSDP        = cnstData.nSDP;
    R1          = sparse(nSDP,nSDP,1,nSDP,nSDP);
    R           = [reshape(R1,nSDP*nSDP,1)',dummy_pag']; 
    b           = c_mul_pAndw_o;
end