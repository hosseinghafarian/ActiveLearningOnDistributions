function [P_p_c, P_p_avg] = learnonfirst_evalonrest(Classification_exp_param, myprofile, learningparams, ...
                                                    TrainSamples, TestSamples)
trainer        = Classification_exp_param.training_func;
tester         = Classification_exp_param.testing_func; 
clmethod       = Classification_exp_param.clmethod;

model          = trainer(learningparams, TrainSamples, true(TrainSamples.n,1));
N_T            = TestSamples.n;
P_p_c          = zeros(N_T, 1);
tstidx         = true(N_T, 1);
callAndcompacc(tester, model);

% for i = 1:N_T
%     tstidx(i)  = true;
%     [inacc]      = callAndcompacc(tester, model);
%     tstidx(i)  = false;
%     P_p_c(i)   = inacc;
% end
P_p_avg = sum(P_p_c)/numel(P_p_c);
    
    function callAndcompacc(tester, model)
       notnoisy       = ~TestSamples.noisy(tstidx);
       n_notnoisy     = sum(notnoisy);
       data_y_test    = TestSamples.Y(tstidx);
       [f_test_val  ] = tester(model, learningparams, TestSamples, tstidx);
%        P_p_c            = max(ones(N_T,1)-f_test_val.*data_y_test', 0); %return loss function value
       y_test         = sign(f_test_val);
       niseq          = bsxfun(@eq, y_test(notnoisy), data_y_test(notnoisy)');
%        niseq          = sum(niseq);
%        P_p_c          = ones(N_T, 1)-niseq/n_notnoisy;
       P_p_c          = ones(N_T, 1)-niseq;
    end 
end