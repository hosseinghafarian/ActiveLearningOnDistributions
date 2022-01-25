function [operators]                   = getConstraints4(learningparams)
global cnstData
if learningparams.scaled_loss
    elements_of_diag = learningparams.c_lambda;
    c_mul_pAndw_o    = learningparams.c_lambda;
else
    elements_of_diag = 1;
    c_mul_pAndw_o    = 1;
end
%% Constraints in A_EC(u) = b_EC;
n_SQ    = 1;                            % # of constraints for sum(q)==1^T batchSize;
n_LE    = 1;                            % # of constraints for Matrix(nSDP,nSDP)==1
n_DL    = cnstData.n_l;                 % # of constraints for diag(Matrix)(initL,initL) + P == 1
n_DUN   = cnstData.n_u;                 % # of constraints for diag(Matrix)(unlab,unlab) + P + q == 1
n_DEX   = numel(cnstData.extendInd);    % # of constraints for diag(Matrix)(extnd,extnd) == q 
n_AEC   = n_SQ + n_LE + n_DL + n_DUN + n_DEX;
n_S     = cnstData.n_S;
Rind    = ones(n_AEC, 2*n_S);
Rvalue  = zeros(n_AEC, 2*n_S);
b_EC    = zeros(n_AEC, 1);
constnum = 5;
A_EC_constranit_func_list = {@eq_AEC_sum_of_q_linearind_val, @eq_AEC_lastelement_of_matrix_linearind_val, ...
                             @eq_AEC_diag_of_labeled_linearind_val, @eq_AEC_diag_of_unlabeled_linearind_val,@eq_AEC_diag_of_extended_linearind_val};
row_ind = 0;
for k = 1:constnum
    [Rind_sq, Rvalue_sq, b_sq, size_of_b_sq, nz_max]  = A_EC_constranit_func_list{k}(c_mul_pAndw_o,elements_of_diag);
    Rind(row_ind+1:row_ind+size_of_b_sq,1:nz_max)                = Rind_sq;
    Rvalue(row_ind+1:row_ind+size_of_b_sq,1:nz_max)              = Rvalue_sq;
    b_EC(row_ind+1:row_ind+size_of_b_sq)                         = b_sq;
    row_ind                                                      = row_ind + size_of_b_sq;
end
const_r  = diag(1:row_ind)*ones(row_ind,2*n_S);
A_EC     = sparse(const_r, Rind(1:row_ind,:), Rvalue(1:row_ind,:), row_ind, cnstData.nConic);
%% Constraints in A_IC(u)<= b_IC
n_SPOUTLIER = 2;                % # of constraints for label noise and outlier
%n_VPQ       = 2*cnstData.n_u;   % # of constraints for r>=G_plus(setunlab,nap+1),r>=-G_plus(setunlab,nap+1)
n_VDUN      = 2*cnstData.n_u;   % # of constraints for v <= a= 1-p-q :=> v+p+c_mul_pAndw_o*q <=1  and -v - a<=0 :=>-v+p+c_mul_pAndw_o*q<=1
n_AIC       = n_SPOUTLIER + n_VDUN;

A_IC_constranit_func_list = {@ineq_AIC_sum_of_p_separate_lbn_outlier_linearind_val,...% @ineq_AIC_abs_of_v_plus_q_and_p_linearind_val, ...
                             @ineq_AIC_abs_of_v_diag_of_unlabeled_linearind_val};
Rind     =  ones(n_AIC, 2*n_S);
Rvalue   = zeros(n_AIC, 2*n_S);
s_IC     = zeros(n_AIC,     1);
constnum = 2;
row_ind = 0;
for k = 1:constnum
    [Rind_sq, Rvalue_sq, b_sq, size_of_b_sq, nz_max]  = A_IC_constranit_func_list{k}(c_mul_pAndw_o,elements_of_diag);
    Rind(row_ind+1:row_ind+size_of_b_sq,1:nz_max)     = Rind_sq;
    Rvalue(row_ind+1:row_ind+size_of_b_sq,1:nz_max)   = Rvalue_sq;
    s_IC(row_ind+1:row_ind+size_of_b_sq)              = b_sq;
    row_ind                                           = row_ind + size_of_b_sq;
end
const_r  = diag(1:row_ind)*ones(row_ind,2*n_S);
Rind(Rind==0&Rvalue==0)=1;
A_IC   = sparse(const_r, Rind(1:row_ind,:), Rvalue(1:row_ind,:), row_ind, cnstData.nConic);                         
%% Constraints in A_EV(u) = b_EV + B_EV \beta
[Rind, Rvalue, b_EV, size_of_b, B_EV] = eq_AEV_vl_y_l_linearind_val(c_mul_pAndw_o,elements_of_diag);
const_r = diag(1:size_of_b)*ones(size(Rind));
n_AEV   = size_of_b;
A_EV    = sparse(const_r, Rind(1:size_of_b,:), Rvalue(1:size_of_b,:), size_of_b, cnstData.nConic); 
%% Constraints in A_IV(u) <= b_IV
[Rind, Rvalue, s_IV, size_of_s, B_IV] = ineq_AIV_abs_of_wo_p_linearind_val(c_mul_pAndw_o,elements_of_diag);
const_r = diag(1:size_of_s)*ones(size(Rind));
n_AIV   = size_of_s;
A_IV    = sparse(const_r, Rind, Rvalue, size_of_s, cnstData.nConic); 
%% Make A_E, A_I, A, B_E, B_I
B_I    = [zeros(n_S,n_AIC),  B_IV];
B_E    = [zeros(n_S,n_AEC),  B_EV];
%stg2 takes more than 6 times of stg
operators = make_transpose_operators();
%% Make Sure Constraint Matrices are independent
make_indep_constraint = false;
if make_indep_constraint
    operators           = make_constraints_indep(operators);
end
operators.A_E           = [operators.A_EC;operators.A_EV];
operators.A_I           = [operators.A_IC;operators.A_IV];
operators.A             = [operators.A_E ;operators.A_I ];
operators.B             = [operators.B_E ;operators.B_I];
operators.B             = sparse(operators.B);
operators.B_V           = [operators.B_EV;operators.B_IV];
%operators.B_x           = [zeros(size(operators.B,1),cnstData.nSDP*cnstData.nSDP),operators.B];
operators.s_I           = [operators.s_IC;operators.s_IV];
operators.b_E           = [operators.b_EC;operators.b_EV];
operators.L_A           = eigs(operators.A*operators.A',1);
operators.L_B           = eigs(operators.B*operators.B',1);
operators.domain_of_x_min_x_u = [-ones(cnstData.nSDP*cnstData.nSDP,1);zeros(cnstData.n_S,1)];
operators.domain_of_x_max_x_u = [ ones(cnstData.nSDP*cnstData.nSDP,1);ones(cnstData.n_S,1) ];
%% Computing cholesky factorization for matrices to be inversed in every iteration of SDP inner Problem.
operators.AA_E  = [ operators.A_EC*operators.A_EC',operators.A_EC*operators.A_EV';operators.A_EV*operators.A_EC',operators.A_EV*operators.A_EV'];
operators.A_EA_I= [ A_EC*A_IC',A_EC*A_IV';A_EV*A_IC',A_EV*A_IV'];
operators.AA_I  = [ operators.A_IC*operators.A_IC',operators.A_IC*operators.A_IV';operators.A_IV*operators.A_IC',operators.A_IV*operators.A_IV'];
%operators.B_EHinvB_I = operators.B_E*cnstData.Hinv*operators.B_I';%operators.B_E*operators.Hinv*operators.B_I'
    function operators                           = make_transpose_operators()
        % Transpose
        operators.n_AEC= n_AEC;
        operators.n_AEV= n_AEV;
        operators.n_AIC= n_AIC;
        operators.n_AIV= n_AIV;
        operators.A_EC = A_EC;
        operators.b_EC = b_EC;
        operators.A_IC = A_IC;
        operators.s_IC = s_IC;
        operators.A_EV = A_EV;
        operators.b_EV = b_EV;
        operators.b_E  = [b_EC;b_EV];
        operators.A_IV = A_IV;
        operators.s_IV = s_IV;
        operators.s_I  = [s_IC;s_IV];
        operators.B_EV = B_EV';
        operators.B_IV = B_IV';
        operators.B_E  = B_E';
        operators.B_I  = B_I';
    end
    function operators                           = make_constraints_indep(operators)
        % check and find independent columns in AB_IV
        % AB_IV       = [A_IV;B_IV];
        % AB_EV       = [A_EV;B_EV];
        % 
        % [AB_IVInd,idxindep]=licols(AB_IV,1e-10); %find a subset of independent columns 
        % % update 
        % A_IV  = AB_IVInd(1:cnstData.nConic,:);
        % B_IV  = AB_IVInd(cnstData.nConic+1:end,:);
        % s_IV  = s_IV(idxindep);
        % n_AIV = size(A_IV,2);
    end
end
%% Commented
% ap(initL) = 1;
% A_IC(:,j) = [zeros(nSDP*nSDP,1)',ap',zeros(2*n,1)'];
% s_IC(j,1)   = n_lbn;
% ap(initL) = 0;
% j = j+1;
% constraint: p>=0
% ap     = zeros(n,1);
% p_ind  = unlabeled;
% ag     = zeros(n,1);
% for k=unlabeled
%     ap(k) = -1;
%     A_IC(:,j)    = [zeros(nSDP*nSDP,1)',ap'];
%     ap(k) = 0;
%     s_IC(j,1)  = 0;
%     j = j+1;
% end
% %cConstraint= [cConstraint,0<=q], where is q<=1? is hidden in p+q<=1,p>=0
% % Constraint :q >=0 == -q <=0 
% aa     = zeros(n,1);
% ap     = zeros(n,1);
% p_ind  = unlabeled;
% ag     = zeros(n,1);
% for k=query
%     R = sparse([nSDP,k],[k,nSDP],[-0.5, -0.5],nSDP,nSDP);
%     A_IC(:,j)    = [reshape(R,nSDP*nSDP,1)',zeros(3*n,1)'];
%     s_IC(j,1)      = 0;
%     j = j+1;
% end