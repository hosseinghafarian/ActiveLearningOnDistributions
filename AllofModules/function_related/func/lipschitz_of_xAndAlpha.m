function [L_x, L_alpha, L_w_obeta, D_x, D_alpha, p_beta_Alpha,L,D,gamma_b]     = lipschitz_of_xAndAlpha(learningparams)
global cnstData
        nSDP        = cnstData.nSDP;
        n_S         = cnstData.n_S;
        L_alpha     = sqrt(n_S)+1/learningparams.lambda*sqrt(cnstData.nap)*nSDP;
        L_w_obeta   = learningparams.lambda_o*cnstData.n_o;
        L_x         = sqrt(n_S)+ L_w_obeta + 1/(2*learningparams.lambda)*nSDP;
        D_x2        = 2*n_S^2 + (1+4*cnstData.batchSize/cnstData.n_q)*n_S+cnstData.batchSize*(cnstData.n_q-2);
        D_x         = sqrt(D_x2);
        D_alpha     = sqrt(n_S);
        delta_x     = 1;
        delta_alpha = 1;
        beta_dxAlpha= L_alpha/L_x * sqrt(D_x*delta_x/(D_alpha*delta_alpha));
        p_beta_Alphaoptimalvalue= 1/(1+beta_dxAlpha);
        p_beta_Alpha= p_beta_Alphaoptimalvalue; 
        L           = sqrt(L_x^2/(p_beta_Alpha*delta_x)+L_alpha^2/((1-p_beta_Alpha)*delta_alpha));
        D           = p_beta_Alpha*D_x + (1-p_beta_Alpha)*D_alpha;
        gamma_b     = L/sqrt(2*D);
end 