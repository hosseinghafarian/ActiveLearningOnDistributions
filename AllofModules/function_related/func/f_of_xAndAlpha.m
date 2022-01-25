function [f]                           = f_of_xAndAlpha(x_k,alpha_k,learningparams,typ)
global cnstData
if nargin == 3, typ =1;end
if typ == 2
        M         = alpha_k*alpha_k';
        f         = alpha_k'*l_of_x(x_k)...
                    -1/(2*learningparams.lambda)*trace(M*(cnstData.KE.*G_of_x(x_k))) ...
                    +eta_of_x(x_k,learningparams)+ learningparams.lambda_o/2*x_k.w_obeta'*cnstData.K_o*x_k.w_obeta;
else
         f         = alpha_k'*l_of_x(x_k)...
                    -1/(2*learningparams.lambda)*alpha_k'*(cnstData.KE.*G_of_x(x_k))*alpha_k ...
                    + eta_of_x(x_k,learningparams) + learningparams.lambda_o/2*x_k.w_obeta'*cnstData.K_o*x_k.w_obeta;
end
end