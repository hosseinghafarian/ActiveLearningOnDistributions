function [model] = leastsquareclassifier(warmStart, x_primalwmst, y_dualwmst, alpha_primalwmst, Model, learningparams)
   global cnstData
   initL     = cnstData.initL';
   lambda   = learningparams.lambda;
   model.dv =0;
   K = cnstData.K(initL,initL);
   model.Klambdainv = inv(K+nL*lambda*eye(nL));
end