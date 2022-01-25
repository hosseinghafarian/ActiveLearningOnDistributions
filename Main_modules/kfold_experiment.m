function [measure] = kfold_experiment(Classification_exp_param, comp_measure, data, myprofile, learningparams)
trainer           = Classification_exp_param.training_func;
tester            = Classification_exp_param.testing_func; 
clmethod          = Classification_exp_param.clmethod;
data.learningparams = learningparams;
kfold     = myprofile.kfold;
global cnstDefs
nfold     = 0;
model     = struct();
measure   = [];
if kfold~=1
   cv        = cvpartition(data.n, 'KFold',kfold);
else
   tridx          = true(data.n,1);
   model          = trainer(learningparams, data, tridx);
   % call cv_tester on data-fold
   if ~isempty(model)
       tstidx = true(data.n,1);
       test_and_contour(tstidx,myprofile.showdata);
       measure = comp_measure(model, acc, measure, true, nfold);
   end
   
   return
end
if isfield(Classification_exp_param,'cmptesting_func')
   cmp_trainer = Classification_exp_param.cmptraining_func;
   cmp_tester  = Classification_exp_param.cmptesting_func;
   comp_model  = true;
else
   comp_model  = false;
end
    for  i=1:kfold
       tridx  = training(cv,i);   
       try 
           ht = tic;
           % call trainer on data-fold
           model  = trainer(learningparams,  data, tridx);
           tstidx = test(cv,i);
           % call cv_tester on data-fold
           if ~isempty(model)
               [acc] = callAndcompacc(tester, model);
               test_and_contour(tstidx,myprofile.showdata);
               if comp_model
                   cmp_model  = cmp_trainer(learningparams,  data, tridx);
                   [acc_comp] = callAndcompacc(cmp_tester, cmp_model);
               else
                   acc_comp   = [];
               end
               measure             = comp_measure(model, acc, measure, false, nfold, acc_comp); 
               nfold = nfold + 1;
           end
           htime = toc(ht);
           %ddisplay_status(mfilename, 6, i, kfold, acc, htime);
       catch ME
           if isfield(measure, 'acc')
               measure.acc= [measure.acc, 0.0];
           else
               measure.acc = 0.0;
           end
           if isfield(measure, 'cmp_acc')
               measure.cmp_acc = [measure.cmp_acc, 0.0];
           else
               measure.cmp_acc = 0.0;
           end
           acc = 0;
           disp('error in trainer or cv_tester in cross_validation');
       end
    end
    measure = comp_measure(model, acc, measure, true, nfold);
    function test_and_contour(tstidx, toshowdata) 
       
       if toshowdata
           if isfield(Classification_exp_param,'cmptesting_func')
               cmp_trainer = Classification_exp_param.cmptraining_func;
               cmp_tester  = Classification_exp_param.cmptesting_func;
               cmp_model   = cmp_trainer(learningparams, data , tridx);
               cmpclmethod = Classification_exp_param.cmpclmethod;
               if isfield(Classification_exp_param,'auxtesting_func')
                   aux_tester  = Classification_exp_param.auxtesting_func;
                   aux_clname  = Classification_exp_param.auxclmethod;
                   testing_func  = {tester, aux_tester, cmp_tester};
                   models        = {model , model     , cmp_model};
                   methodname    = {clmethod, aux_clname, cmpclmethod};
               else
                   testing_func  = {tester, cmp_tester};
                   models        = {model,  cmp_model};    
                   methodname   = {clmethod, cmpclmethod};
               end
           else
                   testing_func  = {tester};
                   models        = {model};    
                   methodname   = {clmethod};
           end
           [lstr]         = get_learning_string(learningparams);
           myprofile.showOptions.title = lstr;
           cnstDefs.fig   = showdata(data, myprofile.showOptions, cnstDefs.fig, testing_func, models, methodname, learningparams);
           [fname] = getfilename_basedon_learningimportant_params(learningparams);
           fname   = [fname,'_',datestr(datetime(),'yy_mm_dd_HH_MM'),'.fig'];
           savefig(cnstDefs.fig, fname);
       end
       % model.noise_detect_accuracy
       
    end
    function [acc] = callAndcompacc(tester, model)
       notnoisy       = ~data.noisy(tstidx);
       n_notnoisy     = sum(notnoisy);
       data_y_test    = data.Y(tstidx);
       data.trsamples = data;
       [f_test_val  ] = tester(model, learningparams, data, tstidx);
       y_test         = sign(f_test_val);
       niseq          = bsxfun(@eq, y_test(notnoisy), data_y_test(notnoisy)');
       niseq          = sum(niseq);
       acc            = niseq/n_notnoisy*100;
    end 
end