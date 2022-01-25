function [learningparams_exp]    = obtain_learning_params(crossvaldata, Classification_exp_param, myprofile)
global cnstDefs
    fn = get_savefilename(false, cnstDefs.result_classification_path , crossvaldata.datasetName, 'OBTAIN_LP', 'ALL' );
    [learning_params_init ]     = get_data_learningsettings(crossvaldata);
    if myprofile.load_lp_from_file && exist(fn,'file')    
       load(fn,'learningparams_exp'); 
       return;
    end
    n_c   = numel(Classification_exp_param);
    learningparams_exp = cell(n_c,1);
    measures_exp       = cell(n_c,1);
    display_status(mfilename,  2, 'finding parameters for classification experiments');
    for i = 1: n_c                                        
        if myprofile.do_cvforobtainlp              
            clexpparam           = Classification_exp_param{i};
            [learningparams_exp{i}, measures_exp{i}] = cross_validation(clexpparam, crossvaldata, myprofile,learning_params_init); 
        else
            learningparams_exp{i}       = learning_params_init;
            measures_exp{i}             = struct();
        end
    end
    if myprofile.save_lp_to_file 
       save(fn,'learningparams_exp', 'measures_exp','crossvaldata','myprofile','Classification_exp_param' );
    end
end