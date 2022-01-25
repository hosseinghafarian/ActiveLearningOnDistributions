function [predict_label, accuracy, decision_values] = svmpredictwrapper_precomputed_kernel(xtest, ytest, model, K)
global cnstDefs
if ~cnstDefs.solver_verbose 
    cmdstr = ' -q ';
else 
    cmdstr = ' ';
end
numtest   = numel(ytest);
Ktilde    = [(1:numtest)', K];
[predict_label, accuracy_seq, decision_values] = svmpredict(ytest', Ktilde, model, cmdstr);
accuracy  = accuracy_seq(1);
end