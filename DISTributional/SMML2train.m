function [model] = SMML2train(learningparams, data, idx )
global cnstDefs
a=tic;
callpre = false;
if callpre
    model = preSMML2train(learningparams, data, idx);
else
    model = newSMML2train(learningparams, data, idx);
end
timet = toc(a);
n_t   = numel(idx);
%fprintf('\n Solving in %d a problem of the size %d', timet, n_t);
end
function [model] = newSMML2train(learningparams, data, idx)
   data_train_y = data.Y(idx);
   Y            = diag(data_train_y);
   distidx      = data.F_id(idx);     % which unique distnums are for training
   traini       = ismember(data.F, distidx); % select instaces which belongs to distributions in idx
   vecdata_train_x = data.X(:, traini);      % instaces which belongs to distributions in idx
   z            = data.Y_L2(traini);
   Z            = diag(z);

   F_a          = data.F(traini);
   n_inst       = numel(F_a);
   n_dist       = numel(distidx);

   assert(data.isTwoLevel, 'For SMML2, data must have two levels!');
   
   K            = data.K(idx,idx);% Here we must update the kernel matrix if kernel params changed = data.K(idx,idx);
   K_L2         = data.K_L2(traini, traini);
   [ A, B] = comp_combkernel(data, idx);
   L            = data.L(traini, traini);
   
   optusingyalmip =false;
   if optusingyalmip
       model = opt_yalSMM2dual(L, A, B, learningparams);
   else
       model = opt_mosekSMM2dual(L, A, B, learningparams);
   end
   function [model] = opt_yalSMM2dual(L, A, B, learningparams)
      C_g      = learningparams.C_g;
      C        = 1/learningparams.lambda;
      C_v      = learningparams.C_v;
      Delta    = learningparams.Delta;
      
      onesinst = ones(n_inst, 1);
      onesdist = ones(n_dist, 1);
      
      yalmip('clear');
      
      alphasdp = sdpvar(n_dist, 1);
      etasdp   = sdpvar(n_dist, 1);
      thetasdp = sdpvar(n_inst, 1);
      simsdp   = sdpvar(n_inst, 1);
      
      cObjective = 0.5*alphasdp'*Y*K*Y*alphasdp + 1/(2*C_g) * simsdp'*L*simsdp;
      cConstraint= [alphasdp >=0, etasdp >=0, thetasdp>=0, ...
                    (sum(alphasdp'*Y)==0):'forbdual', etasdp + alphasdp >= (C + Delta*C)*onesdist, ...
                    thetasdp >= C_v*B'*onesdist, ((etasdp + alphasdp)'*Y*A*onesinst + z*thetasdp == C*onesdist'*Y*A*onesinst):'forbgdual',...
                    simsdp == A'*Y*(etasdp + alphasdp - C* onesdist) + Z*thetasdp];
                
       %opts = sdpsettings('verbose', 0,'solver', 'mosek', 'debug', 1, 'CACHESOLVERS', 1);
       opts = sdpsettings('verbose', 0,'solver', 'mosek', 'debug', 1);
       solve = optimize(cConstraint, cObjective, opts);
       if solve.problem == 0    
            solve.cObjective  = value(cObjective);
            model.w        = Y*value(alphasdp);
            model.b_w      = dual(cConstraint('forbdual'));
            model.b_g      = dual(cConstraint('forbgdual'));
            model.obj_opt  = solve.cObjective;        
            model.name     = 'SMLUPI';
       else
            solve.cObjectivev = -Inf;
            model = struct([]);
            assert(solproblem~=0,'Problem is not solved in function doOptimize in SMML2train');
       end
       
       model.vectrainx   = vecdata_train_x;  
       model.trainy      = data_train_y;
       model.uF_K        = data.F_id(idx);
       model.idxF_K      = idx;
       model.n           = numel(data_train_y);
       model.use_libsvm = false;
   end 
   function [model] = opt_mosekSMM2dual(L, A, B, learningparams)
      C_g      = learningparams.C_g;
      C        = 1/learningparams.lambda;
      C_v      = learningparams.C_v;
      Delta    = learningparams.Delta;
      
      onesinst = ones(n_inst, 1);
      onesdist = ones(n_dist, 1);
%         
%        cConstraint= [alphasdp >=0, etasdp >=0, thetasdp>=0, ...
%                     (sum(alphasdp'*Y)==0):'forbdual', etasdp + alphasdp >= (C + Delta*C)*onesdist, ...
%                     thetasdp >= C_v*B'*onesdist, ((etasdp + alphasdp)'*Y*A*onesinst + z*thetasdp == C*onesdist'*Y*A*onesinst):'forbgdual',...
%                     simsdp == A'*Y*(etasdp + alphasdp - C* onesdist) + Z*thetasdp];
      
      
      % Define the data.
      % First the lower triangular part of q in the objective
      % is specified in a sparse format. The format is:
      %
      % Q(prob.qosubi(t),prob.qosubj(t)) = prob.qoval(t), t=1,...,4
      clear prob % alpha  sim      eta       theta
      n_mosvar = n_dist + n_inst + n_dist  + n_inst;
      Q        = zeros(n_mosvar, n_mosvar);
      Q(1:n_dist, 1:n_dist) = Y*K*Y;
      Q(n_dist+1: n_dist+n_inst , n_dist+1: n_dist+n_inst) = (1/C_g)*L;
      Qtril    = tril(Q);
      [ qosubi, qosubj, qoval ] = find(Qtril);
      % Q
      prob.qosubi = qosubi;
      prob.qosubj = qosubj;
      prob.qoval  = qoval;
      
                % alpha           sim            eta       theta
      lb       = [zeros(n_dist,1);... % alphasdp >=0
                  -Inf(n_inst,1);...  % sim unbounded
                  zeros(n_dist,1);... % etasdp >=0
                  C_v*B'*onesdist];   % thetasdp >= 0 
      ub       = Inf(n_mosvar,1);
      
                % alpha           sim            eta       theta
      a_1      = [ (Y*onesdist)', zeros(1, n_inst+n_dist+n_inst)]; %(sum(alphasdp'*Y)==0):'forbdual'
      lba_1    = 0;                                                
      uba_1    = 0;
      
                % alpha           sim            eta       theta
      a_2      = [ onesinst'*A'*Y, zeros(1, n_inst), onesinst'*A'*Y, onesinst'*Z]; %((etasdp + alphasdp)'*Y*A*onesinst + z*thetasdp == C*onesdist'*Y*A*onesinst):'forbgdual'
      lba_2    = C*onesinst'*A'*Y*onesdist;
      uba_2    = lba_2;
      
                 % alpha           sim            eta       theta
      a_3      = [eye(n_dist), zeros(n_dist, n_inst), eye(n_dist), zeros(n_dist, n_inst)]; % etasdp + alphasdp >= (C + Delta*C)*onesdist
      lba_3    = (C+Delta*C)* onesdist;
      uba_3    = Inf(n_dist,1);
      
      a_4      = [A'*Y, -eye(n_inst), A'*Y, Z];
      lba_4    = C*A'*Y*onesdist;
      uba_4    = lba_4; 
      
%       a_5      = [zeros(n_inst, n_dist), zeros(n_inst, n_inst), zeros(n_inst, n_dist), eye(n_inst)];
%       lba_5    = C_v*B'*onesdist;
%       uba_5    = Inf(n_inst, 1);
%                 a        = [a_1  ; a_2  ; a_3  ; a_4  ; a_5 ];
%       lba      = [lba_1; lba_2; lba_3; lba_4;lba_5];
%       uba      = [uba_1; uba_2; uba_3; uba_4;uba_5];
%       
      
      a        = [a_1  ; a_2  ; a_3  ; a_4  ];
      lba      = [lba_1; lba_2; lba_3; lba_4];
      uba      = [uba_1; uba_2; uba_3; uba_4];
      
      prob.a   = sparse(a);
      % Lower bounds of constraints.
      prob.blc = lba;
      % Upper bounds of constraints.
      prob.buc = uba;
      % Lower bounds of variables.
      prob.blx = sparse(lb);
      % Upper bounds of variables.
      prob.bux = [];
      
      [r,res] = mosekopt('minimize echo(0)', prob);
      if (r==0 || r==10006)
         mosekvar = res.sol.itr.xx;
         model.w = Y* mosekvar(1:n_dist);
         sol = res.sol;
         suc = sol.itr.suc;
         model.b_w      = suc(1);
         model.b_g      = suc(2);
         model.obj_opt  = sol.itr.pobjval;        
         model.name     = 'SMLUPI'; 
         model.solver   = 'DirectMosek';
         model.res      = res;
      else
            solve.cObjectivev = -Inf;
            model.res = res;
            assert(false,'Problem is not solved in function doOptimize in SMML2train');
      end
      model.vectrainx   = vecdata_train_x;  
      model.trainy      = data_train_y;
      model.uF_K        = data.F_id(idx);
      model.idxF_K      = idx;
      model.n           = numel(data_train_y);
      model.use_libsvm = false;
   end
   function [ A, B]  = comp_combkernel(data, idx)
        distidx      = data.F_id(idx);            % Which unique distnums are for training
        traini       = ismember(data.F, distidx); % Select instaces which belongs to distributions in idx        
        
        
%         K            = data.K(idx,idx);% Here we must update the kernel matrix if kernel params changed = data.K(idx,idx);
%         K_L2         = data.K_L2(traini, traini);  
       % L            = K_L2;
        
        A            = zeros(n_dist, n_inst);
        B            = zeros(n_dist, n_inst);
        for fidx     = 1:n_dist
            A(fidx, :) = ismember(F_a, distidx(fidx));
            n_i        = sum(A(fidx, :));
            B(fidx, :) = A(fidx, :) .* (z~=0);
            A(fidx, :) = 1/n_i*A(fidx, :);
%             traini_fidx = find(A(fidx, :));
%             for tidx  = 1:n_dist 
%                 traini_tidx = find(ismember(F_a, distidx(tidx)));
% %                 for i=traini_fidx
% %                     for j= traini_tidx
% %                        L(i, j) = K(fidx, tidx)*K_L2(i, j);
% %                     end
% %                 end    
%             end    
        end    
    end
end

function [model] = preSMML2train(learningparams, data, idx)
    uselibsvm    = false;
    data_train_y = data.Y(idx);
   
    distidx      = data.F_id(idx);     % which unique distnums are for training
    traini       = ismember(data.F, distidx); % select instaces which belongs to distributions in idx
    vecdata_train_x = data.X(:, traini);
    
%     if data.isTwoLevel
%        L2models = MULTSKSVMtrain(data.dataL2.learningparams_init, data.dataL2, distidx);
%     end
    assert(data.isTwoLevel, 'For SMML2, data must have two levels!');
    K            = data.K(idx,idx);% Here we must update the kernel matrix if kernel params changed = data.K(idx,idx);
    K_L2         = data.K_L2(traini, traini);
    [L , E, O]   = comp_combkernel(data, idx);
    n_th         = size(data.K_L2(traini, traini), 1);
    Y_L2         = data.Y_L2(traini);
    lbflg        = data.L2labelflag(traini);
    F            = data.F(traini);
    F_id_tr      = unique(F); % what are the unique distnums 
    assert(numel(F_id_tr)==numel(data_train_y), 'Error in data, inconsistency between number of labels and number of distributions');
    n            = numel(data_train_y);     % size of data 
    Yl           = data_train_y;
    lambda       = learningparams.lambda;
    gamma        = learningparams.gammaL2;
    C_v          = learningparams.C_v;
    Delta        = learningparams.Delta;
    if ~uselibsvm 
        model = optimize_using_yalmip();
        model.use_libsvm = false;
    else
        model = solve_using_libsvm();    
    end
    model.vectrainx   = vecdata_train_x;  
    model.trainy   = data_train_y;
    model.uF_K     = data.F_id(idx);
    model.idxF_K   = idx;
    model.n        = numel(data_train_y);
    
    function model = solve_using_libsvm()
       [cmdstr] = get_libsvm_cmd(learningparams);
       if ~cnstDefs.solver_verbose 
           cmdstr = strcat(cmdstr,' -q ');
       end
       cmdstr     = strcat(cmdstr,' -t 4 ');
       K_indexed   = [(1:n)',K];
       libsvmmodel = svmtrain(data_train_y', K_indexed, cmdstr);
       model.use_libsvm = true;
       model.libsvmmodel = libsvmmodel;
       model.name     = 'SMM_USINGLIBSVM';
    end
    function [model] = optimize_using_yalmip()
    %%  Define YALMIP Variables        
        w            = sdpvar(n,1);
        b_w          = sdpvar(1,1);
        ksi          = sdpvar(n,1);
        betamatvec   = sdpvar(n_th, 1);
        zeta         = sdpvar(n_th, 1);
        gmatvecnew   = L*diag(betamatvec)*E*O;
  %      gmatvec      = comp_g(K, K_L2, F, betamatvec);
        [mug]        = comp_mu_g(gmatvecnew, F);
    %%  Define Problem, Constraints and Objective  
        [solve] = doOptimize();
        if solve.solproblem == 0 
            model.w        = value(w);
            model.b_w      = value(b_w);
            model.obj_opt  = solve.cObjectivev;        
            model.name     = 'SMLUPI';
        else
            model = struct([]);
        end
        function [solve] = doOptimize()
            opts = sdpsettings('verbose', 0,'solver', 'mosek', 'debug', 1, 'CACHESOLVERS', 1);
%             tic;
%             [cConstraint, cObjective] = constraintType();
%             sol = optimize(cConstraint,cObjective, opts);
%             toc
            
            %[primalfeas, dualfeas] = check(cConstraint);
            %solve.solproblem = sol.problem;
            
            tic;
            [cConstraintnew, cObjectivenew] = constraintTypenew();
            
            solnew = optimize(cConstraintnew,cObjectivenew, opts);
            toc
            solve.solproblem = solnew.problem;
            if solve.solproblem == 0    
                solve.cObjectivev = value(cObjectivenew);
            else
                solve.cObjectivev = -Inf;
                assert(solproblem~=0,'Problem is not solved in function doOptimize in SMML2train');
            end
        end
        function [cConstraint, cObjective]    = constraintType()
            C            = 1/lambda;
            Yl           = diag(data_train_y');
            cConstraint  = [];
            cConstr1     = (ksi>=0):'ksi_positive';
            cConstraint  = [cConstraint, cConstr1 ];
            cConstraint  = [cConstraint, ksi + Yl*(mug)>= 0 ];
            cConstraint  = [cConstraint, ksi>= ones(n,1)-Yl*(K*w+b_w*ones(n,1))-Yl*(mug)];
            cConstraint  = [cConstraint, zeta>=0];
            cConstraint  = [cConstraint, zeta>= lbflg'.*(ones(n_th,1)-Y_L2'.*gmatvec')];
            cObjective   = C*sum(Yl*mug+ksi)  + Delta*C*sum(ksi) + C_v*sum(zeta) ...
                           + 1/2*w'*K*w + gamma/2*norm(gmatvec)^2;  
        end
        function [cConstraint, cObjective]    = constraintTypenew()
            C            = 1/lambda;
            Yl           = diag(data_train_y');
            cConstraint  = [];
            cConstr1     = (ksi>=0):'ksi_positive';
            cConstraint  = [cConstraint, cConstr1 ];
            cConstraint  = [cConstraint, ksi + Yl*(mug)>= 0 ];
            cConstraint  = [cConstraint, ksi>= ones(n,1)-Yl*(K*w+b_w*ones(n,1))-Yl*(mug)];
            cConstraint  = [cConstraint, zeta>=0];
            cConstraint  = [cConstraint, zeta>= lbflg'.*(ones(n_th,1)-Y_L2'.*gmatvecnew)];
            cObjective   = C*sum(Yl*mug+ksi)  + Delta*C*sum(ksi) + C_v*sum(zeta) ...
                           + 1/2*w'*K*w + gamma/2*norm(gmatvecnew)^2;  
        end
        function [mug] = comp_mu_g(g_matvec, F)
            F_id   = unique(F);
            assert(numel(F_id)==n, 'Wrong number of distributional examples');
            mug = sdpvar(n, 1);
            for i=1:n
               ind    = (F==F_id(i));
               indnum = find(ind);
               
               n_i    = numel(indnum);
               indg   = ((1:n_th)==indnum(1));
               mug(i) = g_matvec(indg);
               for j= 2:numel(indnum)
                  indg   = ((1:n_th)==indnum(j));
                  mug(i) = mug(i) + g_matvec(indg);
               end
               mug(i) = mug(i)/n_i;
            end
        end
        function [g_matvec ] = comp_g(K, K_L2, F, betaMatvec)
            F_id   = unique(F);
            assert(numel(F_id)==n, 'Wrong number of distributional examples');
            for i=1:n
               ind    = (F==F_id(i)); 
               indnum = find(ind);
               str = sprintf('Computing g_%d_%d:%d\n', F_id(i), min(indnum), max(indnum));
               disp(str);

               ind_i   = F_id==F_id(i);
               for j= 1:numel(indnum) 
                   g_matvec(indnum(j)) = get_g_element(K, K_L2, F, F_id, betaMatvec, indnum(j), ind_i);
               end 
            end
        end
        function [g_i_j] = get_g_element(K, K_L2, F, F_id, betaMatvec, ij, F_i)
            g_i_j = 0;
            for t=1:n
               ind = (F==F_id(t));
               ind_t   = F_id==F_id(t);
               K_Pi_Pt = K(F_i, ind_t);
               g_i_j_t = K_L2(ij, ind)*betaMatvec(ind);
               g_i_j   = g_i_j + K_Pi_Pt*g_i_j_t;   
            end
        end
    end
    function [L, E, O]  = comp_combkernel(data, idx)
        distidx      = data.F_id(idx);            % Which unique distnums are for training
        traini       = ismember(data.F, distidx); % Select instaces which belongs to distributions in idx        
        
        F_a          = data.F(traini);
        n_all        = numel(F_a);
        K            = data.K(idx,idx);% Here we must update the kernel matrix if kernel params changed = data.K(idx,idx);
        K_L2         = data.K_L2(traini, traini);  
        L            = K_L2;
        n_f          = numel(distidx);
        E            = zeros(n_all, n_f);
        O            = ones(n_f, 1);
        for fidx     = 1:n_f
            E(:, fidx) = ismember(F_a, distidx(fidx));
            traini_fidx = find(E(:, fidx))';
            for tidx  = 1:n_f 
                traini_tidx = find(ismember(F_a, distidx(tidx)));
                for i=traini_fidx
                    for j= traini_tidx
                       L(i, j) = K(fidx, tidx)*K_L2(i, j);
                    end
                end    
            end    
        end    
    end
end