function [decision_values] = SVMtester_word2vecseriesappend(model, learningparams, data, idx)
xtest  = data.X_pcaappend(:,idx);
global cnstDefs
cmdstr = ' -b 0 ';
% if ~cnstDefs.solver_verbose 
%     cmdstr = strcat(cmdstr,' -q ');
% end
n   = size(xtest,2);
yt  = zeros(n,1);
[predict_label, ~, decision_values] = svmpredict(yt, xtest', model, cmdstr);
end