function [al_profile ] = make_al_profile(active_profile_id, synthmode, distributional)
   al_profile = set_al_profile();
   
end
function [al_profile] = set_al_profile(varargin)
params = inputParser;
    params.addParameter('maxbatchsize'          ,5               ,@(x) isscalar(x) & x > 0);
    params.addParameter('experiment_num'        ,20             ,@(x) isscalar(x) & x > 0);
    params.addParameter('max_query'             ,50              ,@(x) isscalar(x) & x > 0);
    params.addParameter('batch_size'            ,1               ,@(x) isscalar(x) & x > 0);
    params.addParameter('init_type'             ,5               ,@(x) isscalar(x) & x > 0);
    params.addParameter('n_initial_samples'     ,2               ,@(x) isscalar(x) & x > 0);
    params.addParameter('selType'               ,1               ,@(x) isscalar(x) & x > 0);
    params.addParameter('start_notnoisy'        ,true            ,@(x) islogical(x));
    params.parse(varargin{:});
    al_profile               = params.Results;
end