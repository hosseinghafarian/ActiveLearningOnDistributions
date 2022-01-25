function [SynthesisData, Transductive, showData, TestRatio, SampleRatio, dataset_list] = get_profile(profile, active_profile_id)
        SynthesisData  = profile{active_profile_id}{1}; % data is synthesised or from a dataset file
        Transductive   = profile{active_profile_id}{2}; 
        showData       = profile{active_profile_id}{3};% To show data in 2d experiments or not?
        TestRatio      = profile{active_profile_id}{4};  % what percentage of data will be used for test?
        SampleRatio    = profile{active_profile_id}{5};            % what percentage of data will be used for train?
        dataset_list   = profile{active_profile_id}{6};        
end