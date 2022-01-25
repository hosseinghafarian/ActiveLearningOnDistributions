    function [L_x, L_alpha] = computeLipSchitz(learningparams)
    global cnstData
    %where this computations, come from? 
        normK          = norm(cnstData.KE,'fro');
        L_alpha        = normK/learningparams.lambda;
        L_1            = sqrt(cnstData.nap)*normK/learningparams.lambda;
        L_G            = cnstData.nap*normK/learningparams.lambda;
        L_x            = 2*L_G+2*L_1^2/learningparams.rhox;
    end