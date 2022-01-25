function [cmdstr] = get_libsvm_cmd(learningparams)
lambda = learningparams.lambda;
gamma  = learningparams.KOptions.gamma;
cmdstr = sprintf(' -c  %12.8f ',1/lambda);
gmstr  = sprintf(' -g %12.8f ',gamma);
cmdstr = strcat(cmdstr    ,gmstr);
cmdstr = strcat(cmdstr,' -t 2 -b 0 ');
end