function [R, b, size_of_b]                   = eq_AEC_diag_of_unlabeled(c_mul_pAndw_o,elements_of_diag)
    % this is equivalent to diag(G_{uu})+ p_u + q==1
global cnstData
    t     = 1;
    ap    = zeros(cnstData.n_S,1);
    n_u   = cnstData.n_u;
    R     = spalloc(cnstData.nConic,n_u,2*n_u);
    assert(numel(cnstData.query) == numel(cnstData.extendInd));
    tic;
    b     = zeros(n_u,1);
    for k = 1:cnstData.n_u
       ku        = cnstData.unlabeled(k);
       kq        = cnstData.extendInd(k);
       R1        = sparse([ku,           kq,cnstData.nSDP],...
                          [ku,cnstData.nSDP,           kq],...
                          [1 ,          0.5,          0.5], cnstData.nSDP, cnstData.nSDP);
       ap(ku)    = 1;
       R(:,t)    = [reshape(R1,cnstData.nSDP*cnstData.nSDP,1)',ap'];
       b(t,1)    = elements_of_diag;
       ap(ku)    = 0;
       t = t+1; 
    end 
    size_of_b    = cnstData.n_u;
end