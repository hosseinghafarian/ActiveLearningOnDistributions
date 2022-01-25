function [predict_label, accuracy,decision_values]=libsvm_trainandtest(xtrain, ytrain, xtest, ytest, learningparams)
[cmdstr] = get_libsvm_cmd(learningparams);
% cmdstr = sprintf(' -c %12.8f ',1/lambda);
% gmstr  = sprintf(' -g %12.8f ',gamma);
% cmdstr = strcat(cmdstr    ,gmstr);
% cmdstr = strcat(cmdstr,'-t 2 -b 0 ');

% Train the SVM :cmdstr : '-c 1 -g 0.07 -b 1 -t 4'
model    = svmtrain(ytrain, xtrain', cmdstr);%'-c 1 -g 0.07 -b 1 -t 4' ); % t=4: precomputed kernel, b=1:probability estimates, g:gamma in kernel, c=1/lambda
% Use the SVM model to classify the data
assert(~isempty(model),'libsvm train is not successfull in module libsvm_trainandtest');
[predict_label, accuracy, decision_values] = svmpredict(ytest,xtest', model, '-b 0'); % test the training data
end