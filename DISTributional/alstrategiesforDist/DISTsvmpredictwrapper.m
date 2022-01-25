function [predict_label, accuracy, decision_values] = DISTsvmpredictwrapper(model, learningparams, data, idx)
global cnstDefs
xtest = data.DISTX(:,idx);
ytest = data.Y(idx);
cmdstr = ' -b 0 ';
if ~cnstDefs.solver_verbose 
    cmdstr = strcat(cmdstr,' -q ');
end
[predict_label, accuracy_seq, decision_values] = svmpredict(ytest', xtest', model, cmdstr);
accuracy  = accuracy_seq(1);
end