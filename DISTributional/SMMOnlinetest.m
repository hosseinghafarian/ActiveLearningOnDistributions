function [f_test_val] = SMMOnlinetest(model, learningparams, data, idx)
% distu       = unique(data.F); % what are the unique distnums 
% distidx     = distu(idx);     % which unique distnums are for training
% testi       = ismember(data.F, distidx); % select instaces which belongs to distributions in idx
% data_x_test = data.X(:,testi);
% F           = data.F(testi);
% testFidx    = ismember(unique(F), data.F);
% [KA]        = DISTKernelArray(model, data_x_test, F, learningparams);
global cnstDefs
F_id_test   = data.F_id(idx);
F_id_SV     = model.SV;

[data] = data.data_comp_kernel(data.learningparams, data.Classification_exp_param, data, data.trsamples, F_id_test, F_id_SV);
K_SV_idx = data.K;
model.SMMOnline = true;

[f_test_val] = SMMtest(model, learningparams, data, idx, K_SV_idx);

end

% 
% alpha       = model.alpha;
% n_test      = sum(idx);
% f_test_val  = ones(n_test, 1);
% [data]      = data.data_comp_kernel(data.learningparams, data.Classification_exp_param, data, data.trsamples, F_id_test, F_id_SV);
% KA = data.K;
% 
% onlineclassifier = 2;
% switch onlineclassifier
%     case 1
%         M  = model.M;
%         w  = model.w;
%         
%         for t = 1:numel(F_id_test)
%             k_t  = KA(:,t);  
%             nx_t = M*k_t;
%             f_t  = w*nx_t; 
%             f_test_val(t) = sign(f_t); 
%         end    
%     case 2
%         for t = 1:numel(F_id_test)
%             k_t  = KA(:,t);
%             f_t=alpha*k_t;
%             hat_y_t = sign(f_t);
%             if (hat_y_t==0),
%                 hat_y_t=1;
%             end
%             f_test_val(t) = hat_y_t;
%         end    
% end
% 
% end