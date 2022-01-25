function [R, b, size_of_b]                   = eq_AEC_diag_of_labeled(c_mul_pAndw_o,elements_of_diag)
    % cConstraint= [cConstraint,diag(G_plus(initL,initL))==1-p(initL),...
    %                  diag(G_plus(setunlab,setunlab))==r,...
    % Constraint: diag(G_{ll,uu})==1-g, since g and a are removed it is
    % equivalent to g=p+q , (i.e,diag(G_{ll,uu})==1-(p+q))
    % this is equivalent to diag(G_{ll})+ p_l ==1
global cnstData
    ap    = zeros(cnstData.n_S,1);
    initL = cnstData.initL(cnstData.initL>0)';
    n_l   = numel(initL);
    R     = spalloc(cnstData.nConic,n_l,2*n_l);
    b     = zeros(n_l,1);
    t     = 1;
    for k = initL
        R1         = sparse(k,k,1,cnstData.nSDP,cnstData.nSDP);
        ap(k)      = 1;
        R(:,t)     = [reshape(R1,cnstData.nSDP*cnstData.nSDP,1)',ap']; 
        b(t,1)     = elements_of_diag;
        ap(k)      = 0;
        t          = t+1;
    end
    size_of_b      = numel(initL);
end