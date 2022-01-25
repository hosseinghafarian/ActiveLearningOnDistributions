function [accuracy, querytime, predict_label, model, query_id, compare_result]...
            = ActiveLearner(WMST_beforeStart, al_profile, ModelInfo  ,learningparams ,...
                            TrainSamples, TestSamples, Almethod_funcs, Almethod_params)%jointtrainandquery, querymethodfunc, classifier_func)
%% Main learning and active learning caller. 
% This function call the main multitask active learning
% it calls active learning methods as well as learning method
global cnstData
%TODO: Search Which instance is the optimal instance in this moment. 
%assert(~TrainSamples.isDistData,'data is distributional');
assert(numel(cnstData.query)>=cnstData.batchSize,'there is not enough instances to query');
report_accuracy_BEFORE_addin_query_i = false;
% for deleting any zero number from initL
% method            = al_profile.Querymethod;
% nQueryEachTime    = al_profile.batchSize;
K                 = cnstData.K;
lambda_AL         = learningparams.lambda_AL;
classifier_func   = Almethod_funcs{3};
querymethod_func  = Almethod_funcs{1};
test_func         = Almethod_funcs{4};
joint_train_query = Almethod_funcs{2};
if numel(Almethod_funcs)==8
    [accuracy, predict_label, model, query_id]= do_AL(classifier_func, test_func, joint_train_query, querymethod_func, Almethod_params);
    compare_result.compare = false;
else
    compare = Almethod_funcs{9};
    comp_classifier_func   = compare.Almethod_funcs{3};
    comp_querymethod_func  = compare.Almethod_funcs{1};
    comp_test_func         = compare.Almethod_funcs{4};
    comp_joint_train_query = compare.Almethod_funcs{2};
    [accuracy, predict_label, model, query_id]= do_AL(classifier_func, test_func, joint_train_query, querymethod_func);
    [comp_accuracy, comp_predict_label, comp_model, comp_query_id]= do_AL(comp_classifier_func, comp_test_func, comp_joint_train_query, comp_querymethod_func);
    compare_result.compare    = true;
    compare_result.queryind_1 = query_id;
    compare_result.queryind_2 = query_id;
    compare_result.accuracy_1 = accuracy;
    compare_result.accuracy_2 = comp_accuracy;
    compare_result.predict_label_1 = predict_label;
    compare_result.predict_label_2 = comp_predict_label;
end
%% Active Learning: Calling query method
   function [accuracy, predict_label, model, query_id]= do_AL(trainer, tester, joint_train_query, querymethod, Almethod_params)        
       learningparams.WMST_beforeStart = WMST_beforeStart; 
       querytime = 0;
       if joint_train_query 
            [ALresult, model, Y_set] = trainer(WMST_beforeStart, ModelInfo, learningparams,TrainSamples.X, TrainSamples.Y); 
            if ALresult.active
               query_id  = ALresult.queryind;
               if ~report_accuracy_BEFORE_addin_query_i
                   train_id   = [cnstData.initL(cnstData.initLnozero)',query_id];
               else
                   train_id   = cnstData.initL(cnstData.initLnozero)';
               end
            else
               query_id  = []; 
               train_id  = TrainSamples.F_id;
            end
            y_transductive = TrainSamples.Y;
            indXtrain      = ismember(TrainSamples.F, train_id);
            indtrain       = ismember(TrainSamples.F_id, train_id);
            [predict_label, accuracy, decision_values] = tester(TrainSamples.X(:,indXtrain), TrainSamples.Y(indtrain), TestSamples.X, TestSamples.Y, y_transductive, model, Y_set, learningparams);
       else
            stqtime = tic;
            [query_id]    = querymethod(TrainSamples, cnstData.initL(cnstData.initLnozero), cnstData.query, K, lambda_AL, learningparams, Almethod_params);
            querytime = toc(stqtime);
            % Important:The new query point must be added to the labeled set (initL) in the
            % next time, in accordance with joint_train_query methods. 
            if ~report_accuracy_BEFORE_addin_query_i
               train_id   = [cnstData.initL(cnstData.initLnozero)',query_id];
            else
               train_id   = cnstData.initL(cnstData.initLnozero)';
            end
            idx           = ismember(TrainSamples.F_id, train_id); %unique(TrainSamples.F)(trainind)
            [model]       = trainer(learningparams, TrainSamples, idx);
            [predict_label, accuracy, decision_values] = testing(model, learningparams);            
       end
       assert(~ismember(query_id, cnstData.initL)    ,'error cannot query the same instance again');
       assert( ismember(query_id, TrainSamples.F_id),'Error cannot query an instance which is not in the training set');
       function [predict_label, accuracy, decision_values] = testing(model, learningparams)
         %WARNINING: when there is noisy data, this code behaves completely
         %incorrect. Why?
         notnoisy       = ~TestSamples.noisy;
         n_notnoisy     = sum(notnoisy);
         data_y_test    = TestSamples.Y;
%          uFTest  = unique(TestSamples.F);
%          uFTrain = unique(TrainSamples.F);
         tstidx        = true(TestSamples.n,1);
         [decision_values  ] = tester(model, learningparams, TestSamples, tstidx);
         predict_label  = sign(decision_values);
         niseq          = bsxfun(@eq, predict_label(notnoisy), data_y_test(notnoisy)');
         niseq          = sum(niseq);
         accuracy       = niseq/n_notnoisy*100;
      end
   end
end