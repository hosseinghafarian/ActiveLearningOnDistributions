function [A_IV, s_IV, B_IV ,n_AIV] = ineq_AIV_abs_of_wo_p()
    % Constraint: q <= 1+\phi(X)^T w_o
    % for k = cnstData.extendInd
    %     R = sparse([k,cnstData.nSDP],[cnstData.nSDP,k],[0.5,0.5],cnstData.nSDP,cnstData.nSDP);
    %     A_IV(:,j) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',dummy_pag'];
    %     s_IV(j,1) = 1;
    %     j = j + 1;
    % end
    % %Constraint: q <= 1-\phi(X)^T w_o
    % %A_IV  = [A_IV,A_IV];
    % for k = cnstData.extendInd
    %     R         = sparse([k,cnstData.nSDP],[cnstData.nSDP,k],[0.5,0.5],cnstData.nSDP,cnstData.nSDP);
    %     A_IV(:,j) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',dummy_pag'];
    %     s_IV(j,1)   = 1;
    %     j         = j + 1;
    % end
    % Constraint:  \phi(X)^T w_o  <= p
global cnstData
    ap        = zeros(cnstData.n_S,1);
    j  =1;
    for k   = 1:cnstData.n_S%unlabeled
       ap(k)    = -1;
       A_IV(:,j)= [zeros(cnstData.nSDP*cnstData.nSDP,1)',ap'];
       s_IV(j,1)   = 0;
       ap(k)    = 0;
       j        = j+1;
    end
    % Constraint:  -\phi(X)^T w_o  - p <= 0 
    for k=1:cnstData.n_S%unlabeled
       ap(k)    = -1;
       A_IV(:,j)= [zeros(cnstData.nSDP*cnstData.nSDP,1)',ap'];
       s_IV(j,1)  = 0;
       ap(k)    = 0;
       j        = j+1;
    end
    n_AIV  = j-1;
    Iind     = speye(cnstData.n_S,cnstData.n_S);
    eye_q  = Iind(setdiff(1:cnstData.n_S,cnstData.initL),1:cnstData.n_S);
    % I_rep  = [sparse(eye_q);-sparse(eye_q);speye(cnstData.n_S);-speye(cnstData.n_S)];
    % B_IV   = (I_rep*cnstData.K)';
    I_rep  = [speye(cnstData.n_S);-speye(cnstData.n_S)];
    B_IV   = (I_rep*cnstData.K)';
end