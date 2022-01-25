function [model] = svmtrainwrapper_precomputed_kernel(WMST_beforeStart,  Model, learningparams, x_labeled, y_labeled, K)
global cnstDefs
numtrain = size(K,1);
Ktr      = [(1:numtrain)', K]; 
lambda   =  learningparams.lambda;
str      = sprintf(' %7.4f ',1/lambda);
str      = strcat(' ', str);
cmdstr   = strcat(' -c ',str);
cmdstr   = strcat(cmdstr,' -t 4 ');
if ~cnstDefs.solver_verbose 
    cmdstr = strcat(cmdstr,' -q ');
end
model    = svmtrain(y_labeled', Ktr ,cmdstr);%'-c 1 -g 0.07 -b 1 -t 4' ); % t=4: precomputed kernel, b=1:probability estimates, g:gamma in kernel, c=1/lambda
end