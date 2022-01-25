function [model] = svmtrainwrapper(learningparams, trainsamples, trainind, x, y)
global cnstDefs
% lambda = learningparams.lambda;
% gamma  = learningparams.KOptions.gamma;
% cmdstr = sprintf(' -c  %12.8f ',1/lambda);
% gmstr  = sprintf(' -g %12.8f ',gamma);
% cmdstr = strcat(cmdstr    ,gmstr);
% cmdstr = strcat(cmdstr,' -t 2 -b 0 ');
if nargin == 3
   x_labeled = trainsamples.X(:, trainind);
   y_labeled = trainsamples.Y(trainind);
elseif nargin == 5
   x_labeled = x;
   y_labeled = y;
end
[cmdstr] = get_libsvm_cmd(learningparams);
if ~cnstDefs.solver_verbose 
    cmdstr = strcat(cmdstr,' -q ');
end
if isfield(learningparams, 'probestimate') && learningparams.probestimate
    cmdstr = strcat(cmdstr,' -b 1');
end
% Train the SVM :cmdstr : '-c 1 -g 0.07 -b 1 -t 4'
model  = svmtrain(y_labeled', x_labeled',cmdstr);%'-c 1 -g 0.07 -b 1 -t 4' ); % t=4: precomputed kernel, b=1:probability estimates, g:gamma in kernel, c=1/lambda
end