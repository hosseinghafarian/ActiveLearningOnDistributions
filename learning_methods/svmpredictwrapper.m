function [predict_label, accuracy, decision_values] = svmpredictwrapper(model, learningparams, testsamples, tstidx, x, y)
global cnstDefs
if nargin == 4
   xtest = testsamples.X(:,tstidx);
   ytest = testsamples.Y(tstidx);
else
   xtest = x;
   ytest = y;
end
if isfield(learningparams, 'probestimate') && learningparams.probestimate
    cmdstr = ' -b 1 ';
else
    cmdstr = ' -b 0 ';
end
if ~cnstDefs.solver_verbose 
    cmdstr = strcat(cmdstr,' -q ');
end
[predict_label, accuracy_seq, decision_values] = svmpredict(ytest', xtest', model, cmdstr);
accuracy  = accuracy_seq(1);
end