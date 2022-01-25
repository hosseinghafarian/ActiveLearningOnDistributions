function [active_experiment_sequence] = make_activelearning_experiment(isdist)
global cnstDefs  
cnstDefs.ALMETHOD_ID_TCYBEXPLREPINFLIBSVMPROB 
cnstDefs.ALMETHOD_ID_BMDRACMTRANSKDD
cnstDefs.ALMETHOD_ID_MAED
if ~isdist
        active_experiment_sequence = {...
            {cnstDefs.ALMETHOD_ID_VARLOGPMAL                       ,false, 0, {1}},...
            {cnstDefs.ALMETHOD_ID_TCYBEXPLREPINFLIBSVMPROB         ,false, 0, {}},... 
            {cnstDefs.ALMETHOD_ID_BMDRACMTRANSKDD                  ,false, 0, {}},...
           };

end 
end