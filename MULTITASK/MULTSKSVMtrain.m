function [L2models] = MULTSKSVMtrain(learningparams, dataL2, taskidx)
global cnstDefs
n_t = numel(taskidx);
model = cell(n_t, 1);
tasks = cell(n_t, 1);
for taski = 1:n_t
    traini       = ismember(dataL2.F, taskidx(taski));
    model{taski} = SVMtrain(learningparams, dataL2, traini);
    tasks{taski} = taskidx(taski);
end 
L2models.models  = model;
L2models.tasks   = tasks;
end