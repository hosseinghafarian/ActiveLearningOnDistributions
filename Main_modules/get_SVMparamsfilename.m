function [fname] = get_SVMparamsfilename(datadir, datasetName)
global cnstDefs
str = ['SVM_param_DS=',datasetName,'.mat'];
path  = datadir;

fname = [path,'\',str];
end