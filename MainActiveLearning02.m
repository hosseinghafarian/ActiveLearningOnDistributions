function MainActiveLearning02
%% Main Active Learning Function. 
%% Setting path and Solver
%addpath 'C:\Program Files\Mosek\8\toolbox\r2013a' % for mosek

set_path     = pwd;
setup_paths(set_path);
%% Global variables 
global cnstData
global cnstDefs
global cnstCSVfilePath
global cnstCSVfile
global cnstQTIMECSVfile
global crisisDInd
%cnstCSVfilePath    = 'D:\AL-MAIN\AAAThesis\latex\Papers\Learning From DistributionalPlusVectorial\experiments3\';
%cnstCSVfilePath     = 'D:\AL-MAIN\AAAThesis\latex\Papers\SpectralRobustActiveLearning\time4revise\';
cnstCSVfilePath     = set_path;
cnstCSVfile         = 'w2v-chtrdata';
cnstQTIMECSVfile    = 'time_elapseddataset';
synthmode           = false;%
distributional      = false;
recompute_SVMparams = false;
globaldefinitions(set_path);
%% Methods Definitions and functions list 
[Classification_exp_param]   = make_classification_experiment(distributional);
[almethod_list]              = make_ActiveLearning_methods_list(distributional);
[active_experiment_sequence] = make_activelearning_experiment(distributional);
% list of noise resistant methods from Active learning methods num
%% Profiles: How to run program
logfilename          = sprintf('ActiveLearningMethodslogfile_%s.txt', date()); 
active_profile_id    = getprofileID();
myprofile            = make_profiles(active_profile_id, synthmode, distributional);
al_profile           = make_al_profile(active_profile_id, synthmode, distributional);
do_init_basedon_profile(myprofile);
myprofile.logfileID  = fopen([cnstDefs.main_path,cnstDefs.exceptions_path,'/',logfilename], 'w');
if recompute_SVMparams 
    for datasetId = myprofile.dataset_list
        % load dataset
        [~ ]         = load_dataobject(datasetId, myprofile, recompute_SVMparams);
    end
end
sradataset = false;
if sradataset 
   save_dataset_report('crisis_datasetreport.csv', myprofile); 
end
%% Main loop: Experiment On DataSets
f_it = true; 
for datasetId = myprofile.dataset_list
%      for crisisInd = 1:22
%          crisisDInd = crisisInd;
%         try
            % load dataset
            [data ]                     = load_dataobject(datasetId, myprofile);
            data.dname  = data.datasetName;
            display_status(mfilename,  1, data.datasetName );
            cnstDefs.fig = showdata(data, myprofile.showOptions);

            % do classification experiments using best params obtained4each 
            if myprofile.do_cl
               [measures]               = do_classification_experiment(data, Classification_exp_param, myprofile);
               fn                       = get_savefilename(true, cnstDefs.result_classification_path, data.datasetName,'CLASSIFYFINAL','ALL');
               save(fn,'measures','data','learningparams_exp','myprofile');
            end

            % do active learning experiments ( using best params and parameters? or using just one parameter which is good?) 
            if myprofile.do_al
                % TODO: Split data into train, crossvalidation and test subsets
                [TrainSamples, TestSamples] = selectTrainAndTestSamples(data , myprofile, data.TR_sampling_func, data.TR_settings);
                % obtain learningparams:cross validation, init or loadfromfile
                learningparams_exp          = obtain_learning_params(TrainSamples, Classification_exp_param, myprofile);
                mymethod_id = 1;
                learningparams = learningparams_exp{1};
                if f_it
                   [exp_par_names, exp_pars]  = get_experiment_param_name(almethod_list, active_experiment_sequence);
                   init_qtimemethodname_to_csvfile('Dataset', exp_par_names);
                   f_it = false;
                end
                ACTIVE_LEARNING(active_experiment_sequence, myprofile, al_profile, TrainSamples, TestSamples, learningparams, almethod_list);               
            end
            clear data
%        catch ME
            str = sprintf('failure to work with dataset number %d',datasetId);
            disp(str); 
%        end
%      end
end % dataset     
fclose(myprofile.logfileID);
%% Defining Nested Functions for MainActiveLearning02
    function active_profile_id = getprofileID()
       if ~synthmode
           %active_profile_id    = cnstDefs.CV_CLASSIFICATION;
           %active_profile_id    = cnstDefs.ACTIVELEARNING_SMALLDATASET;
           %active_profile_id    = cnstDefs.RAPIDCV_CLASSIFICATION;
           %active_profile_id    = cnstDefs.FASTCV_CLASSIFICATION;
           %active_profile_id    = cnstDefs.FASTSMALLCV_CLASSIFICATION;
           %active_profile_id    = cnstDefs.TINYCV_CLASSIFICATION;
           active_profile_id    = cnstDefs.ACTIVELEARNING;
           %active_profile_id    = cnstDefs.ACTIVELEARNING_SMALLDATASET;
           %active_profile_id    = cnstDefs.LOADPARAMS_CLASSIFICATION;
        else
           % active_profile_id    = cnstDefs.SYNTHDATA;
           % active_profile_id    = cnstDefs.SYNTHNOACTIVELEARNING;
           %active_profile_id    = cnstDefs.SYNTHDISTDATA;
           %active_profile_id    = cnstDefs.SYNTHSMMPAPERDISTDATA;
           active_profile_id    = cnstDefs.SYNTHSMMPAPERDISTDATA_ALEXP;
        end 
    end
end