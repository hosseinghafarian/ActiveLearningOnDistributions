function [learningparams ] = learning_settings(n, varargin)
%init, gamma, gamma_o, lambda, lambda_o, Ktype, scaled_loss, data_noisy, data_labelnoise
params = inputParser;
    params.addParamValue('gamma'            ,1e-3      ,@(x) isscalar(x) & x > 0);
    params.addParamValue('gamma_o'          ,1e-3      ,@(x) isscalar(x) & x > 0);
    params.addParamValue('gamma_is'         ,1e-3      ,@(x) isscalar(x) & x > 0);
    params.addParamValue('gammaL2'          ,0         ,@(x) isscalar(x) & x >= 0);
    params.addParamValue('C_v'              ,0         ,@(x) isscalar(x) & x >= 0);
    params.addParamValue('Delta'            ,0         ,@(x) isscalar(x) & x >= 0);
    params.addParamValue('BME_thau'         ,0.1       ,@(x) isscalar(x) & x > 0);
    params.addParamValue('varrho'           ,1e-1      ,@(x) isscalar(x) & x >= 0);
    params.addParamValue('kernel_func'      ,@rbf_kernel, @(x)isa(x,'function_handle'));
    params.addParamValue('lambda'           ,1         ,@(x) isscalar(x) & x > 0);
    params.addParamValue('lambda_AL'        ,1         ,@(x) isscalar(x) & x > 0);
    params.addParamValue('sigma_likelihood', 1         ,@(x) isscalar(x) & x > 0);
    params.addParamValue('lambda_o'         ,0.1       ,@(x) isscalar(x) & x > 0);
    params.addParamValue('label_outlier_separate_deal' ,false       ,@(x) islogical(x));
    params.addParamValue('init'             ,struct([]),@(x) isstruct(x));
    params.addParamValue('Ktype'            ,'rbf'     ,@(x) ischar(x)  );
    params.addParamValue('data_noisy'       ,false(1,n),@(x) islogical(x) & length(x) == n);
    params.addParamValue('data_labelnoise'  ,false(1,n),@(x) islogical(x) & length(x) == n);
    params.addParamValue('scaled_loss'      ,false     ,@(x) islogical(x) );
    params.addParamValue('lambda_alpha_D_part'   ,0     ,@(x) isscalar(x) );
    params.addParamValue('lambda_alpha_Q_part'   ,0     ,@(x) isscalar(x) );
    params.addParamValue('rhoalpha'              ,1.0       ,@(x) isscalar(x) & x>=0  );
    params.addParamValue('rhox'                  ,1.0       ,@(x) isscalar(x) & x>=0  );
    params.addParamValue('ca'                    ,1.0       ,@(x) isscalar(x) & x>=0  );
    params.addParamValue('cp'                    ,1         ,@(x) isscalar(x) & x>=0  );
    params.addParamValue('cq'                    ,1.0       ,@(x) isscalar(x) & x>=0  );
    params.addParamValue('AlselectSamplesMethod' ,1.0       ,@(x) isscalar(x) & x>=0  );
    params.addParamValue('AlselectSamplesPercent' ,100       ,@(x) isscalar(x) & x>=0 & x<=100  );
    params.addParamValue('c_o'                   ,1        ,@(x) isscalar(x) & x> 0   );
    params.addParamValue('c_LRJ'                 ,0.2      ,@(x) isscalar(x) & x> 0 &x<=0.5  );
    params.addParamValue('d_REJ'                 ,0.2      ,@(x) isscalar(x) & x> 0 &x<=0.5  );
    params.addParamValue('lambda_gamma'          ,0.2      ,@(x) isscalar(x) & x> 0  );
    params.addParamValue('beta_LRJ'              ,1.1      ,@(x) isscalar(x) & x> 1  );
    params.addParamValue('alpha_LRJ'             ,1        ,@(x) isscalar(x) & x> 0   );
    params.addParamValue('lambda_CaseSpecific'   ,10        ,@(x) isscalar(x) & x> 0   );
    params.addParamValue('noiserate'             ,0.1       ,@(x) isscalar(x) & x> 0   );
    params.parse(varargin{:});
    
par = params.Results;
if ~isempty(par.init)
    learningparams                  = par.init;
    return;
end
learningparams.n          = n;
learningparams.label_outlier_seperate_deal = par.label_outlier_separate_deal;% if it is true, we have onoiseper and lnoiseper fields otherwise we have just n_o
learningparams.isnoisy    = par.data_noisy;
learningparams.isoutlier  = par.data_noisy & ~par.data_labelnoise;
learningparams.lnoiseper  = 100*(sum(learningparams.isnoisy)-sum(learningparams.isoutlier))/numel(learningparams.isnoisy);
learningparams.onoiseper  = 100*sum(learningparams.isoutlier)/numel(learningparams.isoutlier);
learningparams.n_lbn      = floor(n*learningparams.lnoiseper/100)+0.1; % add 0.1 because we want constraints to satisfy slater constraint qualifications.
learningparams.n_o        = floor(n*(learningparams.lnoiseper+learningparams.onoiseper)/100);
learningparams.scaled_loss            = par.scaled_loss;
learningparams.lambda                 = par.lambda;
learningparams.lambda_AL              = par.lambda_AL;
learningparams.sigma_likelihood       = par.sigma_likelihood;
learningparams.lambda_o               = par.lambda_o;
learningparams.lambda_alpha_D_part    = par.lambda_alpha_D_part;
learningparams.lambda_alpha_Q_part    = par.lambda_alpha_Q_part;
learningparams.rhoalpha               = par.rhoalpha;
learningparams.rhox                   = par.rhox;
learningparams.ca                     = par.ca;
learningparams.cp                     = par.cp; 
learningparams.cq                     = par.cq;
learningparams.c_o                    = par.c_o;
learningparams.c_LRJ                  = par.c_LRJ;
par.beta_LRJ                          = 1/(1-2*par.c_LRJ);
learningparams.beta_LRJ               = par.beta_LRJ;
learningparams.alpha_LRJ              = par.alpha_LRJ;
learningparams.lambda_CaseSpecific    = par.lambda_CaseSpecific;
learningparams.AlselectSamplesMethod  = par.AlselectSamplesMethod;
learningparams.AlselectSamplesPercent = par.AlselectSamplesPercent;
learningparams.lambda_gamma           = par.lambda_gamma;
learningparams.varrho                 = par.varrho;
learningparams.BME_thau               = par.BME_thau;
if strcmpi(par.Ktype,'RBF') % 1: Linear, 2: RBF,
    KOptions.KernelType        = 2;
else
    KOptions.KernelType        = 1;
end 
KOptions.kernel_func       = par.kernel_func;
KOptions.gamma             = par.gamma;
KOptions.gamma_o           = par.gamma_o;
KOptions.gamma_is          = par.gamma_is;
KOptions.gamma_r           = KOptions.gamma_is/2;
learningparams.KOptions    = KOptions;
end