function [K, K_o, F_to_ind_tr ] = comp_kernels(TrainSamples, learningparams)
    % Transpose data to make every data columnize
    % And Append one to every x instead of b in w^Tx+b
    % Select Test indices
    [K, K_o, F_to_ind_tr, F_to_ind_te]= get_two_KernelArray(TrainSamples, TrainSamples, learningparams, true);        
end