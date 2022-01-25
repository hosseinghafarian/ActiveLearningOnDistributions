function [operators]                   = getConstraints3(learningparams)
global cnstData
%% This function is comes from testdualanefficientinexact9
dummy_pag = zeros(cnstData.n_S,1);
if learningparams.scaled_loss
    elements_of_diag = learningparams.c_lambda;
    c_mul_pAndw_o    = learningparams.c_lambda;
else
    elements_of_diag = 1;
    c_mul_pAndw_o    = 1;
end
% Constraints in A_EC(u) = b_EC;
j        = 1;  

[A_EC(:,j), b_EC(j,1)]   = eq_AEC_sum_of_q  ();

j        = j+1;
[A_EC(:,j), b_EC(j,1)]   = eq_AEC_lastelement_of_matrix();

j        = j+1;
[R, b, size_of_b]        = eq_AEC_diag_of_labeled();
A_EC(:,j:j+size_of_b-1)  = R;
b_EC(j:j+size_of_b-1,1)  = b;
j        = j+size_of_b;
[R, b, size_of_b]        = eq_AEC_diag_of_unlabeled();
A_EC(:,j:j+size_of_b-1)  = R;
b_EC(j:j+size_of_b-1,1)  = b;
j        = j+size_of_b;

[R, b, size_of_b]        = eq_AEC_diag_of_extended();
A_EC(:,j:j+size_of_b-1)  = R;
b_EC(j:j+size_of_b-1,1)  = b;
j        = j+size_of_b;
n_AEC    = j-1;
%% Constraints in A_IC(u)<= b_IC
%A_IC = sparse(nSDP*nSDP+3*n,7*n_u+2+n_l);
%b_IC = zeros(7*n_u+2,1);
j        = 1;
[R_IC, s, size_of_s]   = ineq_AIC_sum_of_p_separate_lbn_outlier();
A_IC(:,j:j+size_of_s-1)  = R_IC;
A_IC                     = sparse(A_IC);
s_IC(j:j+size_of_s-1,1)  = s;

j        = j+size_of_s;
% the following (commented) constraint function is not correct since using it we cannot
% maintain any relation between diag and corresponding elements v elements
% of SDP matrix. Instead we use the next one
% [R_IC, s, size_of_s]   = ineq_AIC_abs_of_v_plus_q_and_p();
[R_IC, s, size_of_s]     = ineq_AIC_abs_of_v_diag_of_unlabeled();
A_IC(:,j:j+size_of_s-1)  = R_IC;
A_IC                     = sparse(A_IC);
s_IC(j:j+size_of_s-1,1)  = s;
j        = j+ size_of_s;
n_AIC    = j-1;
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
%% Constraints in A_EV(u) = b_EV + B_EV \beta
[A_EV, b_EV, B_EV, n_AEV] = eq_AEV_vl_y_l();
%% Constraints in A_IV(u) <= b_IV
%A_IV = sparse(nSDP*nSDP+3*n,2*n_q+2*n_S);
j    = 1;
[A_IV, s_IV, B_IV ,n_AIV, B_I, B_E] = ineq_AIV_abs_of_wo_p();
A_IV                                = sparse(A_IV);
operators                           = make_transpose_operators();
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
operators.B_x           = [zeros(size(operators.B,1),cnstData.nSDP*cnstData.nSDP),operators.B];
operators.s_I           = [operators.s_IC;operators.s_IV];
operators.b_E           = [operators.b_EC;operators.b_EV];
operators.L_A           = eigs(operators.A*operators.A',1);
operators.L_B           = eigs(operators.B*operators.B',1);
operators.domain_of_x_min_x_u = [-ones(cnstData.nSDP*cnstData.nSDP,1);zeros(cnstData.n_S,1)];
operators.domain_of_x_max_x_u = [ ones(cnstData.nSDP*cnstData.nSDP,1);ones(cnstData.n_S,1) ];
%% Computing cholesky factorization for matrices to be inversed in every iteration of SDP inner Problem.
operators.AA_E  = [ operators.A_EC*operators.A_EC',operators.A_EC*operators.A_EV';operators.A_EV*operators.A_EC',operators.A_EV*operators.A_EV'];
operators.A_EA_I= [ A_EC'*A_IC,A_EC'*A_IV;A_EV'*A_IC,A_EV'*A_IV];
operators.AA_I  = [ operators.A_IC*operators.A_IC',operators.A_IC*operators.A_IV';operators.A_IV*operators.A_IC',operators.A_IV*operators.A_IV'];
%operators.B_EHinvB_I = operators.B_E*cnstData.Hinv*operators.B_I';%operators.B_E*operators.Hinv*operators.B_I'

    function [R, b]                              = eq_AEC_sum_of_q  ()
        % cConstraint= [cConstraint,sum(q)==batchSize];% constraints on q 
        % Constraint :1^T q = bSize
        %n_q       = numel(cnstData.extendInd);
        q_ind     = [repmat(cnstData.nSDP,cnstData.n_q,1),cnstData.extendInd'];  % this is the indexes of q
        R1        = sparse([q_ind(:,1)',q_ind(:,2)'],[q_ind(:,2)',q_ind(:,1)'],repmat(0.5,2*cnstData.n_q,1));
        try 
          R         = [reshape(R1,cnstData.nSDP*cnstData.nSDP,1)',dummy_pag'];
        catch 
          warning('something went wrong here, in eq_AEC_sum_of_q');
        end
        b         = c_mul_pAndw_o*cnstData.batchSize;
    end
    function [R, b]                              = eq_AEC_lastelement_of_matrix()        
        % cConstraint= [cConstraint,G_plus(nap+1,nap+1)==1];
        % Constraint: G(nSDP,nSDP) = 1
        R1          = sparse(cnstData.nSDP,cnstData.nSDP,1,cnstData.nSDP,cnstData.nSDP);
        R           = [reshape(R1,cnstData.nSDP*cnstData.nSDP,1)',dummy_pag']; 
        b           = c_mul_pAndw_o;
    end
    function [R, b, size_of_b]                   = eq_AEC_diag_of_labeled()
        % cConstraint= [cConstraint,diag(G_plus(initL,initL))==1-p(initL),...
        %                  diag(G_plus(setunlab,setunlab))==r,...
        % Constraint: diag(G_{ll,uu})==1-g, since g and a are removed it is
        % equivalent to g=p+q , (i.e,diag(G_{ll,uu})==1-(p+q))
        % this is equivalent to diag(G_{ll})+ p_l ==1
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
    function [Rind, Rvalue, b, size_of_b]                   = eq_AEC_diag_of_labeled_linearind_val()
        % cConstraint= [cConstraint,diag(G_plus(initL,initL))==1-p(initL),...
        %                  diag(G_plus(setunlab,setunlab))==r,...
        % Constraint: diag(G_{ll,uu})==1-g, since g and a are removed it is
        % equivalent to g=p+q , (i.e,diag(G_{ll,uu})==1-(p+q))
        % this is equivalent to diag(G_{ll})+ p_l ==1
        ap    = zeros(cnstData.n_S,1);
        nSDP  = cnstData.nSDP;
        initL = cnstData.initL(cnstData.initL>0)';
        size_of_b      = numel(initL);
        onevec= ones(1,size_of_b);
        rows  = initL; % it is also, equal cols
        vals  = onevec;
        [Rind , Rvalue] = convert_multiple_linear_indices(2, nSDP, nSDP, 1:size_of_b, ...
                                    rows, rows, vals, 1:size_of_b, initL, onevec);
        b     = elements_of_diag*onevec;
    end
    function [Rind, Rvalue, b, size_of_b]                   = eq_AEC_diag_of_unlabeled_linearind_val()
        % this is equivalent to diag(G_{uu})+ p_u + q==1
        n_u   = cnstData.n_u;
        R     = spalloc(cnstData.nConic,n_u,2*n_u);
        nSDP  = cnstData.nSDP;
        neq   = numel(cnstData.query);
        onevec= ones(1,neq);
        halvec= 0.5*onevec;
        
        assert(numel(cnstData.query) == numel(cnstData.extendInd));
        rows  = [cnstData.query,   cnstData.extendInd, repmat(nSDP,1,neq)];  
        cols  = [cnstData.query, repmat(nSDP,1 , neq), cnstData.extendInd];
        vals  = [onevec, halvec, halvec];
        linear_rows = [1:neq,1:neq,1:neq];
        
        [Rind , Rvalue] = convert_multiple_linear_indices(4, nSDP, nSDP, linear_rows, ...
                                    rows, cols, vals, 1:neq, cnstData.query, onevec);
        b     = elements_of_diag*onevec;
        size_of_b    = neq;
    end
    function [R, b, size_of_b]                   = eq_AEC_diag_of_unlabeled()
        % this is equivalent to diag(G_{uu})+ p_u + q==1
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
    function [Rind, Rvalue, b, size_of_b]                   = eq_AEC_diag_of_extended_linearind_val()
        % cConstraint=[cConstraint, diag(G_plus(setQuery,setQuery))==q];
        % Constraint: diag(G_qq)==q
        n_e      = numel(cnstData.extendInd);
        nSDP     = cnstData.nSDP;
        onevec   = ones(1,n_e);
        halvec  = -0.5*onevec;
        rows     = [cnstData.extendInd,   cnstData.extendInd, repmat(nSDP,1,n_e)];  
        cols     = [cnstData.extendInd, repmat(nSDP, 1, n_e), cnstData.extendInd];
        vals     = [onevec, halvec, halvec];
        linear_rows = [1:n_e,1:n_e,1:n_e];
        [Rind , Rvalue] = convert_multiple_linear_indices(4, nSDP, nSDP, linear_rows, ...
                                    rows, cols, vals, 1:n_e, cnstData.query, zeros(n_e,1));
        b         = zeros(n_e,1);
        size_of_b = n_e;    
    end
    function [R, b, size_of_b]                   = eq_AEC_diag_of_extended()
        % cConstraint=[cConstraint, diag(G_plus(setQuery,setQuery))==q];
        % Constraint: diag(G_qq)==q
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
    function [R, s, size_of_s]                              = ineq_AIC_sum_of_p()
        % cConstraint=[cConstraint,sum(p)<=n_o]; %sum(p(initL))<=n_l*lnoiseper/100,
        % Constraint: 1^T p <=n_o
        ap        = ones(cnstData.n_S,1);
        R         = [zeros(cnstData.nSDP*cnstData.nSDP,1)',ap'];
        s         = c_mul_pAndw_o*cnstData.n_o;
        % cConstraint=[cConstraint,sum(p)<=n_o]; %sum(p(initL))<=n_l*lnoiseper/100,
        % Constraint: 1^T p_l <=n_lbn
        assert(cnstData.n_lbn<=cnstData.n_o,'n_lbn, number of noisy labeled points must be less than or equal to n_o, number of noisy labeled points');
        assert(cnstData.n_lbn>0   ,'n_lbn, is zero and so constraint qualifications are not correct and dual problem is not equivalent to primal');
        size_of_s = 1;
    end
    function [R, s, size_of_s]                             = ineq_AIC_sum_of_p_separate_lbn_outlier()
        % cConstraint=[cConstraint,sum(p(all-initL)<=n_u*onoiseper/100]; %sum(p(initL))<=n_l*lnoiseper/100,
        % Constraint: 1^T p(initL) <= n_l*lnoiseper/100, 1^T p(all-initL) <= n_u*onoiseper/100
        ap        = zeros(cnstData.n_S,1);
        initL     = cnstData.initL(cnstData.initL>0);
        ap(initL) = 1;
        R(:,1)    = [zeros(1,cnstData.nSDP*cnstData.nSDP),ap'];
        s(1)      = c_mul_pAndw_o*(cnstData.n_l*cnstData.lnoiseper)/100;
        
        ap        = ones(cnstData.n_S,1);
        ap(initL) = 0; % all ones except for labeled instances which they are in label noise part in the above three lines
        R(:,2)    = [zeros(1,cnstData.nSDP*cnstData.nSDP),ap'];
        s(2)      = c_mul_pAndw_o*(cnstData.n_u*cnstData.onoiseper)/100;
        % cConstraint=[cConstraint,sum(p)<=n_o]; %sum(p(initL))<=n_l*lnoiseper/100,
        % Constraint: 1^T p_l <=n_lbn
%         assert(cnstData.n_lbn<=cnstData.n_o,'n_lbn, number of noisy labeled points must be less than or equal to n_o, number of noisy labeled points');
%         assert(cnstData.n_lbn>0   ,'n_lbn, is zero and so constraint qualifications are not correct and dual problem is not equivalent to primal');
        size_of_s = 2;
    end
    function [R_IC, s, size_of_s]                = ineq_AIC_abs_of_v_plus_q_and_p()
        %cConstraint= [cConstraint,0<=q], where is q<=1? q<=1 is not necessary
        %because of the constraint p+q <=1, all are positive. so, this constraint
        %is deleted. 
        % cConstraint= [cConstraint,r>=G_plus(setunlab,nap+1),r>=-G_plus(setunlab,nap+1)];
        % Constraint: v <= a= 1-p-q :=> v+p+c_mul_pAndw_o*q <=1 
        ap    = ones(cnstData.n_S,1);
        t     = 1;
        for k = 1:cnstData.n_u
           ku        = cnstData.unlabeled(k);
           kq        = cnstData.extendInd(k);
           R         = sparse([ku           , cnstData.nSDP, kq               , cnstData.nSDP     ],...
                              [cnstData.nSDP, ku           , cnstData.nSDP    , kq                ],...
                              [0.5          , 0.5          , 0.5*c_mul_pAndw_o, 0.5*c_mul_pAndw_o ],...
                              cnstData.nSDP,cnstData.nSDP);
           ap(ku)    = 1; 
           R_IC(:,t) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',ap'];
           s(t,1)    = 1;
           ap(ku)    = 0;
           t = t+1; 
        end
        %Constraint: -v - a<=0 :=>-v+p+c_mul_pAndw_o*q<=1
        for k = 1:cnstData.n_u
           ku        = cnstData.unlabeled(k);
           kq        = cnstData.extendInd(k);
           %R         = sparse([ku,cnstData.nSDP,kq,cnstData.nSDP],[cnstData.nSDP,ku,cnstData.nSDP,kq],[-0.5,-0.5,0.5,0.5],cnstData.nSDP,cnstData.nSDP);
           R         = sparse([ku           , cnstData.nSDP, kq               , cnstData.nSDP     ],...
                              [cnstData.nSDP, ku           , cnstData.nSDP    , kq                ],...
                              [-0.5         , -0.5         , 0.5*c_mul_pAndw_o, 0.5*c_mul_pAndw_o ],...
                              cnstData.nSDP,cnstData.nSDP);
           ap(ku)    = 1; 
           R_IC(:,t) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',ap'];
           s(t,1)    = 1;
           ap(ku)    = 0;
           t = t+1; 
        end
        size_of_s    = t-1;
        % %Constraint: p+q<=1
        % for k = 1:n_u
        %    ku        = unlabeled(k);
        %    kq        = query(k);
        %    R         = sparse([kq,nSDP],[nSDP,kq],[0.5,0.5],nSDP,nSDP);
        %    ap(ku)    = 1; 
        %    A_IC(:,j) = [reshape(R,nSDP*nSDP,1)',ap',zeros(2*n,1)'];
        %    s_IC(j,1) = 1;
        %    ap(ku)    = 0;
        %    j = j+1; 
        % end  
    end
    function [Rind , Rvalue, s, size_of_s]       = ineq_AIC_abs_of_v_plus_q_and_p_linearind_val()
        %cConstraint= [cConstraint,0<=q], where is q<=1? q<=1 is not necessary
        %because of the constraint p+q <=1, all are positive. so, this constraint
        %is deleted. 
        % cConstraint= [cConstraint,r>=G_plus(setunlab,nap+1),r>=-G_plus(setunlab,nap+1)];
        % Constraint: v <= a= 1-p-q :=> v+p+c_mul_pAndw_o*q <=1 
        ap       = ones(cnstData.n_S,1);
        n_u      = numel(cnstData.unlabeled);
        n_q      = numel(cnstData.extendInd);
        nSDP     = cnstData.nSDP;
        %% if all of unlabeled samples are not in the query, we deal with two subsets, one that is unlabeled, and candidate for query and one that is unlabeled and not candid for query
        [unlabquery, unlabnotquery] = get_unlabeledAndquery();
        n_uq     = numel(unlabquery);
        n_unq    = numel(unlabnotquery);
        halvecq  = 0.5*ones(1,n_uq);
        halvec   = 0.5*ones(1,n_unq);
        % for instances candid for query
        rows     = [unlabquery           , repmat(nSDP, 1, n_uq),            cnstData.extendInd, repmat(nSDP,1,n_uq)  ];  
        cols     = [repmat(nSDP, 1, n_uq),            unlabquery,   repmat(nSDP,1,n_uq), cnstData.extendInd    ];
        vals     = [               halvecq,                halvecq, halvecq*c_mul_pAndw_o, halvecq*c_mul_pAndw_o];
        linear_rows = [1:n_uq,1:n_uq,1:n_uq,1:n_uq];
        [Rind_pe , Rvalue_pe] = convert_multiple_linear_indices(5, nSDP, nSDP, linear_rows, ...
                                    rows, cols, vals, 1:n_uq, unlabquery, ones(n_uq,1));
        n_pe     = n_uq;
        % for instances not candid for query                        
        rows     = [unlabnotquery           , repmat(nSDP, 1, n_unq)];  
        cols     = [repmat(nSDP, 1, n_unq),            unlabnotquery];
        vals     = [               halvec,                    halvec];
        linear_rows = [1:n_unq,1:n_unq];
        [Rind_pn , Rvalue_pn] = convert_multiple_linear_indices(5, nSDP, nSDP, linear_rows, ...
                                    rows, cols, vals, 1:n_unq, unlabnotquery, ones(n_unq,1));
        n_pn     = n_unq;
        %Constraint: -v - a<=0 :=>-v+p+c_mul_pAndw_o*q<=1
        halvecq  = 0.5*ones(1,n_uq);
        halvec   = 0.5*ones(1,n_unq);
                % for instances candid for query
        rows     = [unlabquery           , repmat(nSDP, 1, n_uq),            cnstData.extendInd, repmat(nSDP,1,n_uq)  ];  
        cols     = [repmat(nSDP, 1, n_uq),            unlabquery,   repmat(nSDP,1,n_uq), cnstData.extendInd    ];
        vals     = [              -halvecq,               -halvecq, halvecq*c_mul_pAndw_o, halvecq*c_mul_pAndw_o];
        linear_rows = [1:n_uq,1:n_uq,1:n_uq,1:n_uq];
        [Rind_ne , Rvalue_ne] = convert_multiple_linear_indices(5, nSDP, nSDP, linear_rows, ...
                                    rows, cols, vals, 1:n_uq, unlabquery, ones(n_uq,1));
        % for instances not candid for query                        
        rows     = [unlabnotquery           , repmat(nSDP, 1, n_unq)];  
        cols     = [repmat(nSDP, 1, n_unq),            unlabnotquery];
        vals     = [               -halvec,                  -halvec];
        linear_rows = [1:n_unq,1:n_unq];
        [Rind_nn , Rvalue_nn] = convert_multiple_linear_indices(5, nSDP, nSDP, linear_rows, ...
                                    rows, cols, vals, 1:n_unq, unlabnotquery, ones(n_unq,1));
        Rind   = [Rind_pe(1:n_pe,:)  ;    Rind_pn(1:n_pn,:);  Rind_ne(1:n_pe,:);  Rind_nn(1:n_pn,:)];
        Rvalue = [Rvalue_pe(1:n_pe,:);  Rvalue_pn(1:n_pn,:);Rvalue_ne(1:n_pe,:);Rvalue_nn(1:n_pn,:)];
        size_of_s = n_pe + n_pn + n_pe + n_pn;
        s      = ones(size_of_s,1);
    end
    function [Rind, Rvalue, s, size_of_s]                = ineq_AIC_abs_of_v_diag_of_unlabeled_linearind_val()
        % In the following r==G_plus(setunlab,setunlab)
        % cConstraint= [cConstraint,r>=G_plus(setunlab,nap+1),r>=-G_plus(setunlab,nap+1)];
        nSDP     = cnstData.nSDP;
        unlab    = cnstData.unlabeled;
        n_u      = numel(unlab);
        oneu     = ones(1,n_u);
        halvec   = 0.5*oneu;
        % Constraint: a= 1-p-q>= v :=> v-a<=0 :=> G_plus(setunlab, nap+1)-G_plus(setunlab,setunlab)<=0 
        rows     = [unlab               , repmat(nSDP, 1, n_u),            unlab];  
        cols     = [repmat(nSDP, 1, n_u),                unlab,            unlab];
        vals     = [             halvec,                halvec,            -oneu];
        linear_rows = [1:n_u, 1:n_u, 1:n_u];
        [Rind_pe , Rvalue_pe] = convert_multiple_linear_indices(5, nSDP, nSDP, linear_rows, ...
                                    rows, cols, vals, 1:n_u, unlab, zeros(1,nu));
        n_pe     = n_u;
        % Constraint: a= 1-p-q>= -v :=> -v-a<=0 :=> -G_plus(setunlab, nap+1)-G_plus(setunlab,setunlab)<=0 
        vals     = [            -halvec,               -halvec,            -oneu];
        [Rind_ne , Rvalue_ne] = convert_multiple_linear_indices(5, nSDP, nSDP, linear_rows, ...
                                    rows, cols, vals, 1:n_u, unlab, zeros(1,nu));
        n_ne      = n_u;
        Rind      = [  Rind_pe(1:n_pe,:);   Rind_ne(1:n_ne,:)];
        Rvalue    = [Rvalue_pe(1:n_pe,:); Rvalue_ne(1:n_ne,:)];
        size_of_s = n_pe + n_ne;
        s         = zeros(size_of_s,1);
    end
    function [R_IC, s, size_of_s]                = ineq_AIC_abs_of_v_diag_of_unlabeled()
        % In the following r==G_plus(setunlab,setunlab)
        % cConstraint= [cConstraint,r>=G_plus(setunlab,nap+1),r>=-G_plus(setunlab,nap+1)];
        % Constraint: a= 1-p-q>= v :=> v-a<=0 :=> G_plus(setunlab, nap+1)-G_plus(setunlab,setunlab)<=0 
        ap        = zeros(cnstData.n_S,1);
        t = 1;
        for k = 1:cnstData.n_u
           ku        = cnstData.unlabeled(k);
           R         = sparse([ku,cnstData.nSDP,ku],[cnstData.nSDP,ku,ku],[0.5,0.5,-1],cnstData.nSDP,cnstData.nSDP);
           R_IC(:,t) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',ap'];
           s(t,1)    = 0;
           t = t+1; 
        end
        % Constraint: a= 1-p-q>= -v :=> -v-a<=0 :=> -G_plus(setunlab, nap+1)-G_plus(setunlab,setunlab)<=0 
        for k = 1:cnstData.n_u
           ku        = cnstData.unlabeled(k);
           R         = sparse([ku,cnstData.nSDP,ku],[cnstData.nSDP,ku,ku],[-0.5,-0.5,-1],cnstData.nSDP,cnstData.nSDP);
           R_IC(:,t) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',ap'];
           s(t,1)    = 0;
           t = t+1; 
        end
        size_of_s    = t-1;
    end
    function [A_EV, b_EV, B_EV, n_AEV]           = eq_AEV_vl_y_l()
        % constraint: v_l = y_l-\Phi(X_l)^T w_o
        % b_EV = Yl-\Phi(X_l)^T w_o
        %A_EV = sparse(nSDP*nSDP+3*n,n_l);
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
    function [A_IV, s_IV, B_IV ,n_AIV, B_I, B_E] = ineq_AIV_abs_of_wo_p()
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
        ap        = zeros(cnstData.n_S,1);
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
        B_I    = [zeros(cnstData.n_S,n_AIC),  B_IV];
        B_E    = [zeros(cnstData.n_S,n_AEC),  B_EV];
    end
    function operators                           = make_transpose_operators()
        % Transpose
        operators.n_AEC= n_AEC;
        operators.n_AEV= n_AEV;
        operators.n_AIC= n_AIC;
        operators.n_AIV= n_AIV;
        operators.A_EC = A_EC';
        operators.b_EC = b_EC;
        operators.A_IC = A_IC';
        operators.s_IC = s_IC;
        operators.A_EV = A_EV';
        operators.b_EV = b_EV;
        operators.A_IV = A_IV';
        operators.s_IV = s_IV;
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