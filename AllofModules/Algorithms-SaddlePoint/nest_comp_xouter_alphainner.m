function [alpha_new, x_next, dualvars,solstatus] = nest_comp_xouter_alphainner(x_0,alpha_0,dualvars0, operators,learningparams,optparams, progress_func,verbose)
    global cnstData
    x_k           = x_0;
    alpha_new     = alpha_0;
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
    x_ap_init     = x_append(x_0);
    y             = [dualvars0.y_EC;dualvars0.y_EV;dualvars0.y_IC;dualvars0.y_IV];
    aug_step      = 1.5;
    opts.max_iter = 100;
    [x_next_app, iter, min_cost] = nesterov_composite_general(@grad_x, @proj_x, x_ap_init, L, opts);
    x_next        = x_decomp_samp(x_next_app, x_k);
    
    function g_x_app  = grad_x(x_app)
        x_dec                 = x_decomp_samp(x_app, x_k);
        %[alpha_new, f_x]     = proxf_alpha(learningparams, optparams, alpha_0, x_dec);
        alphanew  = alpha_0;
        [ f_x, grad_x, res_y] = aug_lagrangian_constraints(y, x_dec, operators, aug_step);
        y_new                 = y + res_y; 
        [f, g_x, g_alpha] = f_xAlpha_grad(x_dec, alpha_new, learningparams);
        fvpAEC            = norm(operators.b_EC-operators.A_EC*x_dec.u);
        fvpAEV            = norm(operators.b_EV-operators.A_EV*x_dec.u-operators.B_EV*x_dec.w_obeta);
        fvpAIC            = norm(x_dec.st(1:n_IC)- operators.A_IC*x_dec.u);
        fvpAIV            = norm(x_dec.st(n_IC+1:end)-operators.A_IV*x_dec.u-operators.B_IV*x_dec.w_obeta);
        vdup              = norm(min(x_dec.st-s_I,0));
        sg                = fvpAEC + fvpAEV + fvpAIC + fvpAIV;
        dify              = norm(y_new-y);
        y                 = y_new;
        g_f_app           = x_append(g_x);
        g_aug_app         = x_append(grad_x);
        g_x_app           = g_f_app + g_aug_app; 
    end
    function y_app = proj_x(x_app, opts_proj)
        x_dec        = x_decomp_samp(x_app, x_k);
        [x_dec]      = project_oncones(x_dec, s_I);
%         [x_dec, c_k] = project_cone_on_domain(x_dec, operators);
        y_app        = x_append(x_dec);
    end
    function [alpha_new, f_x]  = proxf_alpha(learningparams, optparams, alpha0, x_k)
        global KG;
        global h_of_x;
        global alphapref;
        global rhop;
        function [fout,gout]                   = f_lG_x_alpha(alphav)
            %global h_of_x;global alphapref;global rhop;
            fout = -alphav'*h_of_x + 1/2* alphav'*KG*alphav + rhop/(2)*norm(alphav-alphapref)^2;
            gout = -        h_of_x +              KG*alphav + rhop*(alphav-alphapref);
        end
        nap       = cnstData.nap;
        n_S       = cnstData.n_S;
        [l_of_x, G_of_x] = x_obj_get_lG_of_x(x_k);
        rhop      = learningparams.rhoalpha;
        alphapref = alpha0;
        KG        = (cnstData.KE.* G_of_x(1:nap,1:nap)) / lambda;
        h_of_x    = [l_of_x;zeros(nap-n_S,1)];
        tol       = 0.001;
        maxit     = 2000;
        [alpha_new, histout, costdata]  = projbfgs(alpha0,@f_lG_x_alpha,cnstData.up,cnstData.lo,tol,maxit);
        f_x       = costdata(end);
    end
end
