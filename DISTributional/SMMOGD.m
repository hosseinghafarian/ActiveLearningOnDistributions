function [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = SMMOGD(data, options, id_list)
% kernelized online gradient descent algorithm
%--------------------------------------------------------------------------
% Input:
%        Y:    the column vector of lables, where Y(t) is the lable for t-th instance ;
%        K:    precomputed kernel for all the example, i.e., K_{ij}=K(x_i,x_j);
%  options:    a struct containing C, tau, rho, sigma, t_tick;
%  id_list:    a random permutation of the 1,2,...,T;
% Output:
%  classifier:  a struct containing SV (the set of idexes for all the support vectors) and alpha (corresponding weights)
%   err_count:  total number of training errors
%    run_time:  time consumed by this algorithm at a time
%    mistakes:  a vector of online mistake rate
% mistake_idx:  a vector of index, in which every idex is a time and corresponds to a mistake rate in the vector above
%         SVs:  a vector recording the online number of support vectors for every idex in mistake_idx
%         TMs:  a vector recording the online time consumption
%        M_ds:  the number of strong double updating
%        M_dw:  the number of weak double updating
%         M_s:  the number of single updating
%--------------------------------------------------------------------------
%% initialize parameters
Y      = data.Y;
X      = data.X;

t_tick = options.t_tick;
eta = options.eta;
ID = id_list;
err_count = 0;

alpha = [];
SV = [];
mistakes = [];
mistakes_idx = [];
SVs = [];
TMs=[];

%% loop
tic
for t = 1:length(ID),
    id = ID(t);
    %% compute f_t(x_t)
    if (isempty(alpha)),
        f_t=0;
    else
        [data] = data.data_comp_kernel(data.learningparams, data.Classification_exp_param, data, data, id, SV);
        k_t = data.K;
        f_t=alpha*k_t;
    end
    %% count the number of errors
    hat_y_t = sign(f_t);
    if (hat_y_t==0),
        hat_y_t=1;
    end

    y_t=Y(t);
    if (hat_y_t~=y_t),
        err_count = err_count + 1;
    end
    
    if y_t*f_t < 1
        alpha = [alpha eta*y_t];%%%%%%%%
        SV = [SV id];
    end



    %% record performance
    run_time = toc;
    if (mod(t,t_tick)==0),
        mistakes = [mistakes err_count/t];
        mistakes_idx = [mistakes_idx t];
        SVs = [SVs length(SV)];
        TMs=[TMs run_time];
    end
    disp(' ');
end

classifier.SV = SV;
classifier.alpha = alpha;
run_time = toc;

