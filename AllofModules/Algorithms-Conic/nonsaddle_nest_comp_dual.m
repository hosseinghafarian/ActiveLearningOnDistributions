function [alpha_k, x_k, dualvars_k,solstatus] = nonsaddle_nest_comp_dual(x_0,alpha_0,dualvars0, operators,learningparams,optparams, progress_func,verbose)
    global cnstData
    x_k           = x_0;
    zhat          = x_0;
    alpha_k       = alpha_0;
    %% Retriving learningparameters for using in iterations
    c_p           = learningparams.cp;
    c_a           = learningparams.ca;
    lambda        = learningparams.lambda;
    lambda_o      = learningparams.lambda_o;
    rhox          = learningparams.rhox;
    rhoalpha      = learningparams.rhoalpha;
    L             = eigs(cnstData.K,1);
    s_I           = [operators.s_IC; operators.s_IV];
    n_IC          = numel(operators.s_IC);
    b_E           = [operators.b_EC;operators.b_EV];
    [ind_EC, ind_EV, ind_IC, ind_IV, ind_S, ind_Z, ind_v] = dual_vars_indices(operators);
    max_ind       = max([ind_EC,ind_EV,ind_IC,ind_IV,ind_S,ind_Z,ind_v]);
    ind_alpha     = max_ind+1:max_ind+numel(alpha_0);
    dual_ap_init  = dual_append([dualvars0.y_EC;dualvars0.y_EV],[dualvars0.y_IC;dualvars0.y_IV],dualvars0.S,dualvars0.Z,dualvars0.v);
    dual_ap_init(ind_alpha) = alpha_k;
    A             = operators.A;
    normg         = 0;
    [ L_y_E, L_y_I, L_S, L_Z, L_v] = dual_objective_lipschitz(operators);
    L             = max([L_y_E;L_y_I;L_S;L_Z;L_v]);
    y             = [dualvars0.y_EC;dualvars0.y_EV;dualvars0.y_IC;dualvars0.y_IV];
    aug_step      = 1.5;
    opts.max_iter = 500;
    [dual_next_app, iter]    = nesterov_composite_general(@grad_dual, @proj_dual, dual_ap_init, L, opts);
    [y_EC, y_EV, y_IC, y_IV, S, Z , v] = dual_decomp(dual_app, ind_EC, ind_EV, ind_IC, ind_IV, ind_S, ind_Z, ind_v);
    x_k                                = x_conv_from_dual_fullproject(y_EC, y_EV, y_IC, y_IV, S, Z, v, x_0, operators);
    dualvars_k                         = dualvar_conv(y_EC, y_EV, y_IC, y_IV, S, Z ,  v );
    solstatus.converged        = true;%converged;
    solstatus.progressing      = trye;%progressing;
    solstatus.iter             = 1;
    solstatus.rel_gap          = 0;%etagap(iter-1);
    solstatus.conv_meas        = 0;%etaall(iter-1);
    solstatus.dist_to_optimal  = 0;%dist_optimal(iter-1);
    solstatus.dualvar_valid    = true;
    solstatus.primalvar_valid  = true;
    return 
    function g_dual_app = grad_dual(dual_app)
        [y_EC, y_EV, y_IC, y_IV, S, Z , v]    = dual_decomp(dual_app, ind_EC, ind_EV, ind_IC, ind_IV, ind_S, ind_Z, ind_v);
        alpha_k                               = dual_app(ind_alpha);
% how to change value of \hat{z} when alpha changes? in \hat(z) how to
% compute c_alpha?
        c_alpha      = c_of_alpha(alpha_k, learningparams);
        zhat.u         = x_0.u       - c_alpha.u;
        zhat.w_obeta   = x_0.w_obeta - c_alpha.w_obeta;
        zhat.st        = x_0.st      - c_alpha.st; 
        [f_dual, g_y_E, g_y_I, g_S, g_Z, g_v, g_dual_alpha] = dual_objective_split_alpha_grad(b_E,alpha_k, y_EC, y_EV, y_IC, y_IV, S, Z , v, zhat, operators, learningparams, optparams);
        g_dual_app                            = -[dual_append(g_y_E, g_y_I, g_S, g_Z, g_v);g_dual_alpha];
        normg = [normg,norm(g_dual_app)];
    end
    function dual_app = proj_dual(dual_app, opt_proj)
        [y_EC, y_EV, y_IC, y_IV, S, Z , v]    = dual_decomp(dual_app, ind_EC, ind_EV, ind_IC, ind_IV, ind_S, ind_Z, ind_v);
        p_alpha      = dual_app(ind_alpha);
        [p_alpha ] = project_alpha_on_domain(p_alpha);
        Ay           = A'*[y_EC;y_EV;y_IC;y_IV];
        R            = Ay + S + x_0.u;
        [Z,v]        = projon_Conestar(cnstData.extendInd,R, x_0.st,operators.s_IC,operators.s_IV, y_IC,y_IV,cnstData.nSDP,cnstData.n_S);  
        S            = -(Ay+Z+x_0.u);
        S            = proj_oncones(S,cnstData.nSDP,cnstData.n_S,1);   
        dual_app(ind_S) = S;
        dual_app(ind_Z) = Z;
        dual_app(ind_v) = v;
        dual_app(ind_alpha) = p_alpha;
    end
end
