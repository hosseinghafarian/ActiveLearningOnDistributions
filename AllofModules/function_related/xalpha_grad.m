function [c_u, c_beta, c_s, c_alpha]   = xalpha_grad(x_k, alpha_k, x_0, alpha_0, learningparams)
        global cnstData
        nSDP      = cnstData.nSDP;
        n_S       = cnstData.n_S;
        % when h is assumed as diag of G, then we need to change gradient
        % in terms of h not q and  p. Before that we substitute h with 1-q-p
        % since h is part of G. we add gradient term related to h to diag
        % of gradient with respect to G. 
        gprox_x_u   = learningparams.rhox*(x_k.u-x_0.u); % Proximal Gradient term
        gprox_x_beta= learningparams.rhox*(x_k.w_obeta-x_0.w_obeta);
        gprox_x_s   = learningparams.rhox*(x_k.st     -x_0.st);
        gprox_alpha = -learningparams.rhoalpha*(alpha_k-alpha_0);
        
        c_beta      = learningparams.lambda_o*cnstData.K*x_k.w_obeta+gprox_x_beta;
        c_s         = gprox_x_s   ;
        
        Aqq         = zeros(nSDP-n_S-1,1);%-(alpha_k(cnstData.query)+learningparams.ca);  
        A_q         = [zeros(n_S,1);Aqq];
        gh          = learningparams.ca*ones(n_S,1) + alpha_k(1:n_S);
        gH          = diag([gh;zeros(nSDP-1-n_S,1)]);
        
        gU          = [gH-1/(2*learningparams.lambda)*(cnstData.KE).*(alpha_k*alpha_k'),A_q;A_q',0];
        gp          = learningparams.cp*ones(n_S,1);
        c_u         = [reshape(gU,nSDP*nSDP,1);gp]+gprox_x_u;
        
        c_alpha     = l_of_x(x_k)-1/learningparams.lambda*(cnstData.KE.*G_of_x(x_k))*alpha_k-gprox_alpha;
        
        c_beta      = -c_beta;
        c_s         = -c_s;
        c_u         = -c_u;
end