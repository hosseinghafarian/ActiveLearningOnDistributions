function ACTIVE_LEARNING(active_experiment_sequence, myprofile, al_profile, TrainSamples, TestSamples, learningparams, almethod_list) 
    global WMST_appendDataInd
    global WARMStart
    global WMST_beforeStart 
    global cnstData
    global cnstDefs
    % Select Active Learning and Classifier Methods for Experiments

    [~, method_store_inds, name_store_inds] = all_of_storeinds(active_experiment_sequence, almethod_list);
       %{cnstDefs.ALMETHOD_ID_ALGORITHMROBUSTAL, false, 0},...

    WARMStart           = true;
    WMST_beforeStart    = true;
    WMST_appendDataInd  = [];
    batchSize           = al_profile.batch_size;
    maxexp_num          = al_profile.experiment_num;
    max_query_round_num = min(al_profile.max_query,min(TrainSamples.n_al, TrainSamples.n));
    max_method_num      = numel(almethod_list);
    maxbatchsize        = al_profile.maxbatchsize;
    zeromaxbatchsize    = zeros(maxbatchsize,1);
    zero_queryrnd_expnum= zeros(max_query_round_num,maxexp_num,max_method_num);
    initL               = cell(max_method_num ,max_query_round_num);  
    assert(batchSize <=maxbatchsize);
    make_initL_accuracy_zero();
    % Loop for methods on datasetId  
    n_experiment        = numel(active_experiment_sequence);
    max_ind             = 0;
    [exp_par_names, exp_pars]  = get_experiment_param_name(almethod_list, active_experiment_sequence);

    [K, K_o, KA, KA_o, F_to_ind_row, F_to_ind_col] = setup_Kernels(learningparams); % it changes TrainSamples and TestSamples
    TrainSamples.K      = K;
    TrainSamples.K_o    = K_o;
    TrainSamples.F_to_ind = F_to_ind_row; % which K(i,*) belongs to F_to_ind dist or instance
    cnstDataSet(K, K_o, KA, KA_o, F_to_ind_row, F_to_ind_col, learningparams, al_profile, TrainSamples);
   % Loop to repeat Experiment for computing Average Performance
    repeatexp  = 1;
    maxexp_num = 10;
    while repeatexp    <= maxexp_num 
        % Model and Data: Kernel Matrix, lambda ,C  
        initL_all          = selectInitialSamples(al_profile, TrainSamples);        
        initL{repeatexp,1} = initL_all;
        %% Set Global Data that doesn't change during Active Learning loop.  
        for experiment_seq_i = 1:n_experiment 
            cnstDataInitLset(initL{repeatexp,1}, learningparams, al_profile, TrainSamples);
            [~, method_ind, Almethod_funcs, experimentdatafile] ...
                     = setup_AL_Method(almethod_list, active_experiment_sequence{experiment_seq_i}, TrainSamples, learningparams);
            [exp_success, query_round_numi] = ActiveLearningLoop(Almethod_funcs, exp_pars{experiment_seq_i}, experiment_seq_i, repeatexp, maxexp_num);
            if query_round_numi > max_ind, max_ind = query_round_numi; end
            save_or_retry();
        end
        repeatexp = repeatexp + 1;
    end
    ACC_PLOT_Data  = computeAverage(accuracy, maxexp_num);
    Time_Plot_Data = computeAverage(timespent, maxexp_num);
    Plotfilename   = get_filename(TrainSamples.datasetName, learningparams.KOptions.gamma, learningparams.KOptions.gamma_o, learningparams.lambda, learningparams.lambda_o);
%    save([cnstDefs.main_path,cnstDefs.result_activelearning_path,Plotfilename,'.mat'])%save all of the variables especially:,'ACC_PLOT_Data');
    figfilename = [cnstDefs.main_path,cnstDefs.result_activelearning_path,Plotfilename];
    my_export_plot_AL(exp_par_names, figfilename, ACC_PLOT_Data, TrainSamples.datasetName);    
%     my_export_plot_AL(exp_par_names, ['time-', Plotfilename], Time_Plot_Data, TrainSamples.datasetName, 'Number of queries', 'Query Time');    

    n_methods = numel(exp_par_names);
    timeavgmethod = sum(Time_Plot_Data,2)/size(Time_Plot_Data,2);
    append_qtimedata_to_csvfile(TrainSamples.dname, timeavgmethod(1:n_methods));
    
    disp('Hi');
    function [success, query_round_numi]    = ActiveLearningLoop(Almethod_funcs, Almethod_params, accstoreidx, repeatexp, exp_num)
        TestSamples.K     = KA;
        TestSamples.F_to_ind_row = cnstData.F_to_ind_row;
        active_retry_num  = 5;
        pre_qu_numi       = -1;
        failure_repeat    = 0;
        %          = Almethod_funcs{5};
        max_query_round_num  = al_profile.max_query;
        ModelInfo.model      = 0; % only to remember we have model 
        title_mod         = 10;
        % Call Active Learner02 Before Main Active Learning loop
        WMST_beforeStart  = true;
        str               = sprintf('Starting Active Learning Method:%s, experiment number:%3d of%3d\n',Almethod_funcs{6},repeatexp,exp_num);
        disp(str);
        title_of_data     = sprintf(' Exp# |  Q# | #labeled | #qremaind | Accuracy | AL Method');
        disp(title_of_data);
        success           = true;           
        %% Main Active Learning Loop
%             query_round_num   = min(floor(numel(cnstData.query)/cnstData.batchSize),max_query_round_num);
        query_round_numi  = 0;
        budget_left       = min(50, min(numel(cnstData.query), TrainSamples.n_al));
        while budget_left > 0 && success  
          %try 
               [acc_ML, querytime, ~, ModelInfo.model,queryInstance, ~]...
                  = ActiveLearner(WMST_beforeStart, al_profile, ModelInfo, learningparams,...
                              TrainSamples, TestSamples, Almethod_funcs, Almethod_params); 
               query_round_numi = query_round_numi + 1; 

               WMST_beforeStart = false;
               % Store Performance Results 
               storePerfRes(acc_ML);
               budget_left      = budget_left      - numel(queryInstance(queryInstance>0));
               %output
               show_data(myprofile, TrainSamples, initL{repeatexp});
               display_progress();
          %catch ME
             %error_handler();
          %end
        end
        sc_str = 'successfull';
        if ~success, sc_str = 'failed'; end 
        str = sprintf('End of %s experiment number:%3d of ALMethod:%s\n',sc_str, repeatexp, Almethod_funcs{6});
        disp(str);
        function display_progress()
            if mod(query_round_numi, title_mod)==0
                  disp(title_of_data);
            end
            str = sprintf(' %3d | %3d  | %4d | %4d | %7.4f  | %s', repeatexp, query_round_numi, sum(cnstData.initL>0), budget_left, acc_ML(1), Almethod_funcs{6});
            disp(str);   
        end
        function error_handler()
                error_catch(ME);
                if pre_qu_numi == query_round_numi
                    failure_repeat = failure_repeat + 1;
                    if failure_repeat >= active_retry_num
                       success = false;
                    end
                    error_msg = ME.message;
                    if failure_repeat == 1 
                       str_e   = sprintf('Query round:%3d of method:%s, Failed:%s, retrying the same query\n',query_round_numi, Almethod_funcs{6}, error_msg);
                    elseif success
                       str_e   = sprintf('Query round:%3d of method:%s, Failed:%s, retry #%3d of the same query\n',query_round_numi, Almethod_funcs{6}, error_msg, failure_repeat);
                    else
                       str_e   = sprintf('Query round:%3d of method:%s, Failed:%s, retry failed #%3d times, aborting this experiment\n',query_round_numi, Almethod_funcs{6}, error_msg, failure_repeat);   
                    end
                    disp(str_e);
                    fprintf(logfileID, str_e);
                end
                pre_qu_numi = query_round_numi;
        end
        function storePerfRes(acc_ML)
            accuracy(query_round_numi,repeatexp,accstoreidx)     = acc_ML;
            timespent(query_round_numi,repeatexp,accstoreidx)    = querytime;
            %sizeinitLset(query_round_numi,repeatexp,accstoreidx) = size(initL{repeatexp},2);
            initL{repeatexp,query_round_numi+1}       = queryInstance;
            [initLcurrnexInd] = cnstDataUpdate(queryInstance);
            failure_repeat =  0; % active learning call success
            pre_qu_numi    = -1; % pre repeatedly failed active learning query call 
        end
        function error_catch(ME)
           saveinfofilename = sprintf('var_%s_%s_%d_%d.mat', data.datasetName, Almethod_funcs{6}, repeatexp, query_round_numi); 
           strError = sprintf('Error in ActiveLearner for method %s on dataset %s in repeat %d, round %d \n, data(variables) stored in file %s, Exception info stored in %s_excep\n', ...
                              Almethod_funcs{6}, data.datasetName,  repeatexp, query_round_numi, saveinfofilename);
           save([cnstDefs.main_path, cnstDefs.exceptions_path,'/',saveinfofilename]);
           saveinfofilename = sprintf('var_%s_%s_%d_%d_Excep.mat', data.datasetName, Almethod_funcs{6}, repeatexp, query_round_numi); 
           ErrorMsg = ME.message;
           funcstrerror = '';
           for k=1:length(ME.stack)
               funcerror = ME.stack(k);
               funcstrerror = sprintf('%s line %d of function %s in file %s\n', funcstrerror, funcerror.line, funcerror.name, funcerror.file);
           end
           error_log = sprintf('%s, Error details: %s in %s\n', strError, ErrorMsg, funcstrerror);
           save([cnstDefs.main_path, cnstDefs.exceptions_path,'/',saveinfofilename],'ME');
           % save to log file
           w = warning ('off','all');
           fprintf(logfileID, error_log);
           warning(w);
        end    
    end
    function [K, K_o, KA, KA_o, F_to_ind_row, F_to_ind_col] = setup_Kernels(learningparams)
        if TrainSamples.isDistData
           [TrainSamples.DISTX, TrainSamples.DISTK, TrainSamples.DISTind] = comp_DISTAVG(TrainSamples, learningparams); 
           TrainSamples.invDISTKlam = inv(TrainSamples.DISTK+learningparams.lambda*eye(size(TrainSamples.DISTK)));
           [TestSamples.DISTX , TestSamples.DISTK , TestSamples.DISTind ] = comp_DISTAVG(TestSamples, learningparams);
           
        end
        [K, K_o,  F_to_ind_row]    = comp_kernels(TrainSamples, learningparams);
        [KA, KA_o,~, F_to_ind_col] = get_two_KernelArray(TrainSamples, TestSamples, learningparams, false);% Kernel between train and test samples,last parameter is recomp
    end
    function make_initL_accuracy_zero
        accuracy     = zero_queryrnd_expnum;
        timespent    = zero_queryrnd_expnum;
        %sizeinitLset = zero_queryrnd_expnum;
        for i=1:maxexp_num 
            for j=1:max_query_round_num
                initL{i,j}= zeromaxbatchsize; 
            end
        end     
    end
    function save_or_retry()
       if exp_success 
            %save([cnstDefs.main_path,cnstDefs.result_activelearning_path, experimentdatafile]);%save all of the variables ,'accuracy','sizeinitLset','initL', 'TrainSamples','TestSamples','','','' );
            str_e     = sprintf('Experiment number %d of method %s, successfully performed. Results saved in file %s\n', ...
                                 repeatexp, Almethod_funcs{6}, experimentdatafile);
            fprintf(myprofile.logfileID, str_e);
       else
            str_e   = sprintf('Attempt to solve experiment failed, repeating experiment...\n');
            fprintf(myprofile.logfileID, str_e);
       end 
    end
end