function [R, b]                              = eq_AEC_sum_of_q  (c_pwo,elements_of_diag)
% cConstraint= [cConstraint,sum(q)==batchSize];% constraints on q 
% Constraint :1^T q = bSize
%n_q       = numel(cnstData.extendInd);
global cnstData
dummy_pag = zeros(cnstData.n_S,1);
    nSDP      = cnstData.nSDP;
    n_q       = cnstData.n_q;
    one2q     = ones(1,2*n_q);
    q_ind     = [repmat(nSDP,n_q,1),cnstData.extendInd'];  % this is the indexes of q
    R1        = sparse([q_ind(:,1)',q_ind(:,2)'],[q_ind(:,2)',q_ind(:,1)'],repmat(0.5,2*n_q,1));
    R         = [reshape(R1, nSDP*nSDP,1)',dummy_pag'];
    b         = c_pwo * cnstData.batchSize;
end