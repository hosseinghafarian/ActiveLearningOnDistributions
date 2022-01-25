function [dualvars_k, x_next, solstatus] = lssdp_NesterovComp_primal_noscale(max_conv_meas, max_rel_gap, max_iter, objectivefunc,x_0, dualvars_0, operators,learningparams,optparams)
%%   
%   In this function we solve argmin_x f(x)+\Psi(x), 
%      where f(x)= \rho/2 \Vert A*x.u+B*x.w_obeta-[b_E;x.st]\Vert^2 and 
%            \Psi(x) = strong_cvx_mu/2 \Vert x-x_0\Vert^2
global cnstData
s_I    = [operators.s_IC;operators.s_IV];
       q_ind = [repmat(cnstData.nSDP,cnstData.n_q,1),cnstData.extendInd']; 
       qinds = sub2ind([cnstData.nSDP,cnstData.nSDP],[q_ind(:,1);q_ind(:,2)],[q_ind(:,2);q_ind(:,1)]); 
       G   = sdpvar(cnstData.nSDP, cnstData.nSDP);
       p   = sdpvar(cnstData.n_S,1);
       u   = [reshape(G,cnstData.nSDP*cnstData.nSDP,1);p];
       w_obeta = sdpvar(cnstData.n_S,1);
       st      = sdpvar(size(s_I,1),1);
       x_var.u = u;
       x_var.w_obeta = w_obeta;
       x_var.st      = st;
       cConstraint = [G>=0, p>=0, u(qinds)>=0, x_var.u<=operators.domain_of_x_max_x_u, x_var.u>= operators.domain_of_x_min_x_u];

[x_0]  = project_oncones(x_0, s_I);
rel_dist_func = objectivefunc.dist_x_opt;
proof_checker = objectivefunc.ProofChecker;
f_func        = @f_exp_substitute_of_constraints;% f_ls_substitute_of_constraints;
Psi_func      = @Psi_func_xconic_distx_x_0;
dist_func     = @euclidean_dist_of_x;
 
% Setting starting values of variables
x_next           = x_0;
v_k_x         = x_0;
x_pre         = x_next;
n_I           = operators.n_AIC + operators.n_AIV;
% Retriving learningparameters for using in iterations
psi_strcvx = 5;% strong convexity parameter for Psi(x)
lipsch_f_of_x = optparams.L_x;
L             = 5000;% lipschitz constant of function f(x) above. It must be computed using operators.
% Exactness and max iteration parameters
max_iterADMM  = optparams.stmax_iterADMM;
convergemeasure = zeros(max_iter,3);
L_0    = 10000;
[T_x ] = dualGradientMethod(x_0,L_0);
solstatus.converged  = converged;
solstatus.progressing= progressing;
solstatus.iter       = iter - 1;
solstatus.rel_gap          = realmax;
solstatus.conv_meas        = convergemeasure(iter-1,1);
solstatus.dist_to_optimal  = dist_optimal(iter-1);
solstatus.dualvar_valid    = false;
solstatus.primalvar_valid  = true;
return 
    function accelerated_Composite()
        % Starting loop    
        A_pre         = 1;

        g_acc_x       = get_null_x(n_I);
        acc_f         = 0; 
        acc_g_inner_x = 0;
        iter          = 1;
        converged     = false;
        progressing   = true;
        max_iter      = 100;
        while ~converged && iter <= max_iter && progressing
            % Update A_k
            a_k       = 1+psi_strcvx*A_pre+ sqrt((1+psi_strcvx*A_pre)^2+4*lipsch_f_of_x*(1+psi_strcvx*A_pre)*A_pre)/lipsch_f_of_x;
            A_curr    = A_pre + a_k;
            % Computing v_k_x
            [v_k_x]   = v_k_min(g_acc_x, x_0, A_curr, operators, learningparams, optparams);                                
            [is_in_v_k]  = isincones(v_k_x, s_I);
            % update x_k 
            update_v_k_x_dual();
            % compute gradient
            [ f_ls_x(iter), c_y] = f_ls_substitute_of_constraints(x_curr, operators);
            normg = x_norm(c_y);
            % compute gradient step
            [x_next ]       = T_L(x_curr, c_y, x_0, L, operators, learningparams, optparams);
            [is_in_x_k]  = isincones(x_next, s_I);
            % update g_acc_x as accumlator of gradient function 

            % updating performance measures
            [acc_f, acc_g_inner_x, g_acc_x,normg2] = linear_approximator_update(x_next, a_k, acc_f, acc_g_inner_x, g_acc_x, f_func, Psi_func, dist_func, operators);
            [checked, Ineq ] = proof_checker(f_func, Psi_func, dist_func, x_0, x_next, v_k_x, iter, A_curr, a_k, acc_f, acc_g_inner_x, g_acc_x, operators, psi_strcvx);
            convergemeasure(iter,1) = norm(x_next.u-x_pre.u)+norm(x_next.st-x_pre.st)+norm(x_next.w_obeta-x_pre.w_obeta);
            convergemeasure(iter,2) = norm(c_y.u) + norm(c_y.w_obeta) + norm(c_y.st);
            if convergemeasure(iter,1) <= max_conv_meas
                converged = true;
            end
            % Prepare for the next iterate
            x_pre                = x_next;    
            % update Nesterov's Coeff
            A_pre                = A_curr;
            learningparams.rhox  = learningparams.rhox*optparams.mul;
            iter                 = iter + 1;
        end           
    end
    function [T_x ] = dualGradientMethod(x_0,L_0)
        gamma_u       = 2;
        gamma_d       = 2;
        g_acc_x       = get_null_x(n_I);
        acc_f         = 0; 
        acc_g_inner_x = 0;
        L             = L_0;
        a_k           = 1/L;
        A_k           = 0;
        T_xpre        = x_0;
        v_k           = v_k_min(g_acc_x, x_0, a_k, operators, learningparams, optparams); 
        [acc_f, acc_g_inner_x, g_acc_x,normg2] = linear_approximator_update(v_k, a_k, acc_f, acc_g_inner_x, g_acc_x, f_func, Psi_func, dist_func, operators);
        L_d(1)        = L;
        for k = 2: max_iter
            [T_x, M_k] = GradientIteration(v_k, L, gamma_u);     
            [phi_val(k), grad_f_x, subgrad_psi_x ] = phi_func(T_x, x_0, f_func, Psi_func, operators, psi_strcvx); 
            a_k        = 1 / M_k;  
            L          = max(L_0, M_k/gamma_d);
            L_d(k)     = L;
            [acc_f, acc_g_inner_x, g_acc_x,normg2] = linear_approximator_update(v_k, a_k, acc_f, acc_g_inner_x, g_acc_x, f_func, Psi_func, dist_func, operators);
            v_k        = v_k_min(g_acc_x, x_0, a_k, operators, learningparams, optparams); 
            [checked, Ineq ] = proof_checker(f_func, Psi_func, dist_func, x_0, T_x, v_k, k, A_k, a_k, acc_f, acc_g_inner_x, g_acc_x, operators, psi_strcvx);
            A_k        = A_k + a_k;
            convergence(k)= euclidean_dist_of_x(T_x, T_xpre);
            T_xpre     = T_x;
        end
    end
    function [T_y, L] =  GradientIteration(y, M,gamma_u)
        L    = M;
        [T_y, m_val]  = T_L_m_L(y, x_0, L, operators, learningparams, optparams, psi_strcvx);
        [phi_val, grad_f_x, subgrad_psi_x ] = phi_func(T_y, x_0, f_func, Psi_func, operators, psi_strcvx); 
        while  phi_val > m_val
            L                                   = L*gamma_u;
            [T_y, m_val]                        = T_L_m_L(y, x_0, L, operators, learningparams, optparams, psi_strcvx);
            [phi_val, grad_f_x, subgrad_psi_x ] = phi_func(T_y, x_0, f_func, Psi_func, operators, psi_strcvx); 
        end
    end
    function [x_next, obj_val]   = argmin_gx_psix_normx_y(g, x_0, psi_mul, y, rho_y, operators, learningparams, optparams)
       cObjective  = x_inner(g,x_var) ;
       cObjective  = cObjective + psi_mul*Psi_func(x_var, x_0, operators, psi_strcvx);
       cObjective  = cObjective + rho_y/2*dist_func(x_var,y);
       sol         = optimize(cConstraint,cObjective);
       assert(sol.problem==0,'cannot solve subproblem');
       x_next.u    = value(x_var.u);
%        assert((x_next.u-operators.domain_of_x_max_x_u)<=0,'infeasible solve!');
%        assert((x_next.u-operators.domain_of_x_min_x_u)>=0,'infeasible solve!');
       x_next.w_obeta = value(x_var.w_obeta);
       x_next.st      = value(x_var.st);
       obj_val        = value(cObjective);
    end
    function [x_next, obj_val ]  = T_L(y, c_y, x_0, L, operators, learningparams, optparams)
             psi_mul    = 1;
             rho_y      = L;
             [x_next, obj_val]   = argmin_gx_psix_normx_y(c_y, x_0, psi_mul, y, rho_y, operators, learningparams, optparams);
    end
    function [T_x, m_val]  = T_L_m_L(y, x_0, L, operators, learningparams, optparams, psi_strcvx)
        [f_y, g ]              = f_func(y, operators);
        [T_x, obj_val]         = T_L(y, g, x_0, L, operators, learningparams, optparams);
        [m_val, f_y, grad_f_y] = m_L(y, T_x, x_0, L, f_func, Psi_func, dist_func, operators, psi_strcvx);
    end
    function [v_k_x]  = v_k_min(g_acc_x, x_0, a_k, operators, learningparams, optparams)
             y          = x_0;
             rho_y      = 1;
             [v_k_x]    = argmin_gx_psix_normx_y(g_acc_x, x_0, a_k, y, rho_y, operators, learningparams, optparams);
    end
    function [v_k_x,dualvars]  = solveBCGDDualProblem(x_k,dualvars, x_G,c_u,c_beta,c_s,unscoperators,learningparams,optparams,arho,mu_reg)
            
            gamma_reg      = arho+mu_reg; 
            Ghat.u         = (1/gamma_reg)*(-c_u    + arho*x_G.u      + mu_reg*x_k.u);
            Ghat.w_obeta   = (1/gamma_reg)*(-cnstData.Qinv*c_beta + arho*x_G.w_obeta+ mu_reg*x_k.w_obeta);
            Ghat.st        = (1/gamma_reg)*(-c_s    + arho*x_G.st     + mu_reg*x_k.st);
            
            gscale         = 1;
            operators      = unscoperators;
            %[gscale,operators] = scaleProblem(learningparams,unscoperators,Ghat);
            
            s_I            = [operators.s_IC;operators.s_IV];
            star           = 1; 
            y_ECtil        = dualvars.y_EC;
            y_EVtil        = dualvars.y_EV;
            y_ICtil        = dualvars.y_IC;
            y_IVtil        = dualvars.y_IV;
            Stil           = dualvars.S   ;

            soltype        = 1;
            %% Step 1
            Ay             = operators.A_EC'* y_ECtil + operators.A_IC'* y_ICtil + operators.A_EV'* y_EVtil + operators.A_IV'* y_IVtil;
            [Xapprox,p,q,qyu]     = getu_Parts(Ay); 
            R              = Ay + Stil + Ghat.u;
            [Xapprox,p,q,qyu]     = getu_Parts(R); 
            [Z,v]          = projon_Conestar(cnstData.extendInd,R, x_G.st,operators.s_IC,operators.s_IV, y_ICtil,y_IVtil,cnstData.nSDP,cnstData.n_S);
            [Xapprox,p,q,qyu]     = getu_Parts(Z);  
            [y_EC,y_EV]    = x_proxLmu_y_E(soltype,optparams.tol4LinearSys,optparams.maxit4LinearSys,...
                                           Z ,  v, Stil, y_ECtil, y_EVtil, y_ICtil, y_IVtil,...
                                           Ghat, operators);

            [y_IC,y_IV]    = x_proxLmu_y_I(soltype,optparams.tol4LinearSys,optparams.maxit4LinearSys,...
                                           Z ,  v, Stil, y_EC, y_EV, y_ICtil, y_IVtil,...
                                           Ghat, operators);

            Ay             = operators.A_EC'*y_EC + operators.A_IC'*y_IC + operators.A_EV'*y_EV + operators.A_IV'*y_IV;                     
            [Xapprox,p,q,qyu]     = getu_Parts(Ay); 
            S              = -(Ay+Z+Ghat.u);
            S              = proj_oncones(S,cnstData.nSDP,cnstData.n_S,star);   
            [Xapprox,p,q,qyu]     = getu_Parts(S); 
 
            [y_IC,y_IV]    = x_proxLmu_y_I(soltype,optparams.tol4LinearSys,optparams.maxit4LinearSys,...
                                           Z ,  v, S, y_EC, y_EV, y_IC, y_IV,...
                                           Ghat, operators);
            Ay             = operators.A_EC'*y_EC + operators.A_IC'*y_IC + operators.A_EV'*y_EV + operators.A_IV'*y_IV;
            [Xapprox,p,q,qyu]     = getu_Parts(Ay); 
            [y_EC,y_EV]    = x_proxLmu_y_E(soltype,optparams.tol4LinearSys,optparams.maxit4LinearSys,...
                                           Z ,  v, S, y_EC, y_EV, y_IC, y_IV,...
                                           Ghat, operators);
            %% 
            Ay             = operators.A_EC'*y_EC + operators.A_IC'*y_IC + operators.A_EV'*y_EV + operators.A_IV'*y_IV;  
            [Xapprox,p,q,qyu]     = getu_Parts(Ay);
            Xp             = proj_oncones(Ghat.u+(Ay+S+Z),cnstData.nSDP,cnstData.n_S,0);   
            [Xapprox,p,q,qyu]     = getu_Parts(Xp); 
            Y              = proj_onP(Ghat.u+(Ay+S),cnstData.nSDP,cnstData.n_S,cnstData.extendInd);
            [Xapprox,p,q,qyu]     = getu_Parts(Y); 
            v_k_x.st       = gscale*min(Ghat.st-[y_IC;y_IV],s_I);
            v_k_x.w_obeta  = gscale*(Ghat.w_obeta + cnstData.Qinv*(operators.B_EV'*y_EV+operators.B_IV'*y_IV));
            %v_k_x.u        = gscale*proj_oncones(Ghat.u+(Ay+Z),cnstData.nSDP,cnstData.n_S,0);   
            v_k_x.u        = gscale*proj_oncones(Ghat.u+(Ay+S+Z),cnstData.nSDP,cnstData.n_S,0); 
            
            dualvars.y_EC  = y_EC;
            dualvars.y_EV  = y_EV;
            dualvars.y_IC  = y_IC;
            dualvars.y_IV  = y_IV;
            dualvars.S     = S;        
    end    
    function update_v_k_x_dual()
        x_curr.u           = (A_pre * x_next.u      + a_k * v_k_x.u      ) / A_curr;
        x_curr.st          = (A_pre * x_next.st     + a_k * v_k_x.st     ) / A_curr;
        x_curr.w_obeta     = (A_pre * x_next.w_obeta+ a_k * v_k_x.w_obeta) / A_curr;     
    end
end
