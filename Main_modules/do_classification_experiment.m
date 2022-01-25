function [measures]    = do_classification_experiment(data, Classification_exp_param, myprofile)
global cnstDefs
data.Classification_exp_param = Classification_exp_param;

standardcv = true;
parallelcv = false;
myfastcv   = false;
if standardcv
    if parallelcv  
       crossvalfunc = @cross_validationparallel;
    else   
       crossvalfunc = @cross_validation; 
    end
else    
   if myfastcv 
       crossvalfunc = @myfastcrossval; % This method of mine may be wrong
   else    
       crossvalfunc = @fastCrossValidation_jmlr2015Krueger; 
   end
end
global cnstCSVfile
cnstCSVfile = [cnstCSVfile, '-', data.datasetName,'.csv'];
        % obtain learningparams:cross validation, init or loadfromfile
    learningparams_exp          = obtain_learning_params(data, Classification_exp_param, myprofile);
    n_c          = numel(Classification_exp_param);
    nrep         = 10; %just for test
    measure_list = cell(n_c,1);
    func_names   = cell(n_c,1);
    clmethods    = cell(n_c,1);
    %% for each classifier
    init_methodname_to_csvfile('#of training points', Classification_exp_param);
    tr = 1;
    train_ratio = 0.9;
    
    
    %for train_ratio = 0.05:0.05:0.8
        myprofile.trainratio  = train_ratio;
        myprofile.testratio   = 1- train_ratio;
        crossvaltrain_ratio = 0.6;
        crossvaltest_ratio  = 0.2;
        mycrossvalprofile           = myprofile;
        mycrossvalprofile.trainratio= crossvaltrain_ratio;
        mycrossvalprofile.testratio = crossvaltest_ratio;
        
        trainsize             = floor(data.n*train_ratio);
        for i = 1: n_c             
            trainer             = Classification_exp_param{i}.training_func;
            tester              = Classification_exp_param{i}.testing_func; 
            learning_params_init = learningparams_exp{i};
            func_names{i}       = func2str(trainer);
            clmethods{i}        = Classification_exp_param{i}.clmethod;
            %% repeat the following experiments nrep times 
            accuracyrep         = zeros(nrep, 1);
            precisionrep        = zeros(nrep, 1);
            recallrep           = zeros(nrep, 1);
            specificityrep      = zeros(nrep, 1);
            f1scorerep          = zeros(nrep, 1);
            if isfield(Classification_exp_param{i},'cmptesting_func')
               cmp_trainer = Classification_exp_param{i}.cmptraining_func;
               cmp_tester  = Classification_exp_param{i}.cmptesting_func;
               compmode  = true;
            else
               compmode  = false;
            end
            mainniseq = [];
            compniseq = [];
            strmsg = sprintf('Function %s, Please wait...', func_names{i}); 
            h = waitbar(0,strmsg);
            savedlp = cell(nrep, 1);
            savedlpfile = ['save_lp_',func_names{i},'.mat'];
            for rep=1:nrep
                fprintf('Repeat %d of experiment', rep);
                % do cross validation to obtain learningparams
                [TrainSamplescrv, TestSamplescrv] = selectTrainAndTestSamples(data , mycrossvalprofile, data.TR_sampling_func, data.TR_settings);
                [learningparams, measures] = crossvalfunc(Classification_exp_param{i}, TrainSamplescrv, myprofile, learning_params_init);
                
                clear TrainSamplescrv TestSamplescrv
                % TODO: Split data into train, crossvalidation and test subsets
                [TrainSamples, TestSamples] = selectTrainAndTestSamples(data , myprofile, data.TR_sampling_func, data.TR_settings);
                % learn model: training using training data and learningparams    
                si = 0;
                while(si <=5)
                    try
                        [accuracyrep(rep), precisionrep(rep), recallrep(rep), specificityrep(rep), f1scorerep(rep)]...
                           = train_test(trainer, tester, false);
                       si = 6;
                    catch ME
                        si = si + 1;
                        if si>5
                            assert(false, 'cannot train and test for computing accuracy. May be the size of the train data is so large');
                        end
                    end
                end    
                testmeasures.accuracy = accuracyrep(rep);
                testmeasures.precision = precisionrep(rep);
                testmeasures.recall    = recallrep(rep);
                testmeasures.specificity = specificityrep(rep);
                testmeasures.f1score     = f1scorerep(rep);
                savedlp{rep} = learningparams;
                savedlp{rep}.testmeasures = testmeasures;
                save(savedlpfile, 'savedlp');
                if compmode 
                    [accuracyrep_cmp(rep), precisionrep_cmp(rep), recallrep_cmp(rep), specificityrep_cmp(rep), f1scorerep_cmp(rep)]...
                       = train_test(cmp_trainer, cmp_tester, true);
                end                
                if compmode
                   tposmain = mainniseq.true_y<0;
                   predpostmain = (mainniseq.pred_y<0)&tposmain;
                   fneqcomp = compniseq.pred_y>0;
                   predneqcomp = fneqcomp&tposmain;
                   maintposcompfneq = predpostmain & predneqcomp;
                   indlog = find(maintposcompfneq);
                   textoftrueposfalseneq = TestSamples.textof_tweets(indlog);
                   save('textforpaper.mat', 'textoftrueposfalseneq');
                end
                waitbar(rep / nrep, h);
                fprintf('End of repeat %d of experiment', rep);
            end
            close(h);
            measure = compute_clmeasure(accuracyrep, precisionrep, recallrep, specificityrep, f1scorerep);
            measure_list{i}             = measure;
            if compmode
                measure_cmp = compute_clmeasure(accuracyrep_cmp, precisionrep_cmp, recallrep_cmp, specificityrep_cmp, f1scorerep_cmp);
                measure_list{i}.measure_cmp = measure_cmp;
            end


            % save resultr
            fn = get_savefilename(true, cnstDefs.result_classification_path, TrainSamples.datasetName,'CLASSIFICATION',clmethods{i});
            save(fn, 'measure_list','TrainSamples','myprofile','learningparams_exp');
        end
        %% prepare to output results
        [trmeasures{tr}]        = array_of_struct_to_struct_of_array(measure_list,{'acc_avg','acc_std'});
        
        append_data_to_csvfile(trainsize, trmeasures{tr});
        tr = tr+ 1;
    %end    
    measures.experiments_details = measure_list;
    measures.func_names          = func_names;
    measures.clmethods           = clmethods;
    measures.data                = data;
    function [accuracyrep, precisionrep, recallrep, specificityrep, f1scorerep] = train_test(trainer, tester, compmode)
       model = training(trainer, compmode);
        % compute accuracy of model on test data
       data_y_test    = TestSamples.Y; 
       y_test_prd      = testing(tester, model, learningparams, compmode);
        if ~compmode 
            mainniseq.eq    = bsxfun(@eq, y_test_prd, data_y_test');
            mainniseq.true_y= data_y_test';
            mainniseq.pred_y= y_test_prd;
        else
            compniseq.eq    = bsxfun(@eq, y_test_prd, data_y_test');
            compniseq.true_y= data_y_test';
            compniseq.pred_y= y_test_prd;
        end
        [accuracyrep, precisionrep, recallrep, specificityrep, f1scorerep] ...
                        = compute_classification_metrics(y_test_prd, data_y_test'); 
    end
    function y_test = testing(tester, model, learningparams, comparemode)
       TestSamples.learningparams = learningparams;
       if ~comparemode
           if ~isfield(Classification_exp_param{i}, 'comp_kernel') || Classification_exp_param{i}.comp_kernel
               [TestSamples] = TestSamples.data_comp_kernel(learningparams, Classification_exp_param, TestSamples, TrainSamples);
           end
       end
       TestSamples.trsamples = TrainSamples;
       [f_test_val  ] = tester(model, learningparams, TestSamples, true(TestSamples.n,1));
       y_test         = sign(f_test_val);
    end
    function model = training(trainer, comparemode)  
        learningparams.use_secondkernel = false;
        if ~comparemode
            if ~isfield(Classification_exp_param{i}, 'comp_kernel') || Classification_exp_param{i}.comp_kernel
                 [TrainSamples] = TrainSamples.data_comp_kernel(learningparams, Classification_exp_param, TrainSamples);
            end
        end
        model          = trainer(learningparams, TrainSamples, true(TrainSamples.n,1));
   end
end