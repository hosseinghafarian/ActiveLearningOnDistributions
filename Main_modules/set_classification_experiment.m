function [par] = set_classification_experiment(methodname, training_func,  testing_func, make_learning_list, comp_kernel,...
                                                           cmpmethodname,  cmptraining_func, cmptesting_func,...
                                                           auxmethodname,  auxtesting_func, varargin)
params = inputParser;
    params.addParameter('two_kernel' ,false           ,@(x) islogical(x));
    params.parse(varargin{:});
    
    par               = params.Results;                                                       
    par.clmethod      = methodname;
    par.training_func = training_func;
    par.testing_func  = testing_func;
    par.making_learning_list_func = make_learning_list;
    par.comp_kernel   = comp_kernel;
if nargin >= 8
    par.cmpclmethod      = cmpmethodname;
    par.cmptraining_func = cmptraining_func;
    par.cmptesting_func  = cmptesting_func;
end
if nargin == 9
    par.auxclmethod      = auxmethodname;
    par.auxtesting_func  = auxtesting_func;
end
    
end