function [model] = HILBsvmtrainwrapper(learningparams, data, ind)
global cnstDefs
x_labeled = data.DISTX(:,ind);
y_labeled = data.Y(ind);
% lambda = learningparams.lambda;
% gamma  = learningparams.KOptions.gamma;
% cmdstr = sprintf(' -c  %12.8f ',1/lambda);
% gmstr  = sprintf(' -g %12.8f ',gamma);
% cmdstr = strcat(cmdstr    ,gmstr);
% cmdstr = strcat(cmdstr,' -t 2 -b 0 ');
[cmdstr] = get_libsvm_cmd(learningparams);
if ~cnstDefs.solver_verbose 
    cmdstr = strcat(cmdstr,' -q ');
end
% Train the SVM :cmdstr : '-c 1 -g 0.07 -b 1 -t 4'
model  = svmtrain(y_labeled', x_labeled',cmdstr);%'-c 1 -g 0.07 -b 1 -t 4' ); % t=4: precomputed kernel, b=1:probability estimates, g:gamma in kernel, c=1/lambda
end