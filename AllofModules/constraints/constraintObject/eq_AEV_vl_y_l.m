function [A_EV, b_EV, B_EV, n_AEV]           = eq_AEV_vl_y_l(c_mul_pAndw_o)
    % constraint: v_l = y_l-\Phi(X_l)^T w_o
    % b_EV = Yl-\Phi(X_l)^T w_o
    %A_EV = sparse(nSDP*nSDP+3*n,n_l);
global cnstData
dummy_pag = zeros(cnstData.n_S,1);
    initL = cnstData.initL(cnstData.initL>0)';
    j     = 1;
    for k = initL
       R           = sparse([k,cnstData.nSDP],[cnstData.nSDP,k],[0.5,0.5],cnstData.nSDP,cnstData.nSDP);
       A_EV(:,j)   = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',dummy_pag'];
       b_EV(j,1)   = c_mul_pAndw_o*cnstData.Yl(k);
       j = j+1;
    end
    n_AEV    = j-1;
    Iind     = speye(cnstData.n_S,cnstData.n_S);
    I_l      = Iind(cnstData.initL(cnstData.initL>0),1:cnstData.n_S);
    B_EV     = (I_l*cnstData.K)'; 
end