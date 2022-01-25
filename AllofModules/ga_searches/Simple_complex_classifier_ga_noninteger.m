function [model] = Simple_complex_classifier_ga_noninteger(learningparams, data, idx) 
    data_train_x = data.X(:,idx);
    data_train_y = data.Y(idx);
    data_train_noisy  = data.noisy(idx);
    
    [K, K_o ]      = get_two_KernelMatrix(data_train_x, learningparams.KOptions);
    n              = numel(data_train_y);
    Y              = diag(data_train_y);
    lambda_o       = learningparams.lambda_o;    
    lambda         = learningparams.lambda;
    [ wg, w_og, fval] = simple_complex_gasearch(data_train_y, K, K_o, lambda, lambda_o, learningparams.cp);
    %[ wg, w_og, fval] = simple_complex_pso_search(data_train_y, K, K_o, lambda, lambda_o, learningparams.cp);
    
    objga          = objective_func([wg;w_og]');
    model.trainx   = data_train_x;  
    model.trainy   = data_train_y;
    model.n        = numel(data_train_y);

    model.w        = abs(wg)';
    model.w_obeta  = w_og;
    model.obj_opt  = fval;

    model.w_oxT_i  = w_og*K_o;
    model.p        = abs(model.w_oxT_i);
    model.data_train_noisy = data_train_noisy;
    model.name     = 'Simple-Complex_ga_noninteger';
    function yval = objective_func(x)
            Y   = diag(data_train_y);
            w   = x(1:n)';
            w_o = x(n+1:2*n)';
            yval   = sum((1-K_o*Y*w_o).*max(1-K*Y*w,0)) + lambda/2*w'*K*w + lambda_o/2*w_o'*K*w_o + learningparams.cp*sum(abs(K_o*w_o));
    end
end