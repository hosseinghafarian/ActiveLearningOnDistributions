function [ALmethod_id, method_ind, Almethod_funcs, experimentdatafile]...
           = setup_AL_Method(almethod_list, experiment_sequence, TrainSamples, learningparams)
%% Setup method method_ind : Active Learning Method  
% setup active learning and classifier method, batchsize and etc.       
        [Almethod_funcs, method_ind, found] = get_method_funcs(almethod_list, experiment_sequence{1});
        assert(found,'method not found');
        experimentdatafile   = get_exp_filename(almethod_list{method_ind}{6}, TrainSamples, learningparams); 
        ALmethod_id           = Almethod_funcs{7};
        compare.compare       = experiment_sequence{2};
        if compare.compare
           compare.ALmethod       = experiment_sequence{3};
           [compare.Almethod_funcs, ~, cmpfound] = get_method_funcs(almethod_list, compare.ALmethod);
           if ~cmpfound, error('method for compare not found');end
           Almethod_funcs{numel(Almethod_funcs)+1} = compare;
        end
end