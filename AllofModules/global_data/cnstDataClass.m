classdef cnstDataClass
    properties
        K
        Q
        n_l
        n_u
        n_q
        n_o
        n_lbn
        n_S      % use all of labeled+unlabeled data
        n        % size of alphatild ( alpha + tau)
        query    % indecies of cnstData.queryidx points
        nSDP    % size of the SDP Matrix 
        nConic
        batchSize
        initL
        unlabeled
        KE
        KEn
        KEp
        KEvec
        Kuvec 
        KQinv
        Yl
    end
    methods
        function cnstDataSet(ModelInfo,batchSizein,n_oin,initL,unlabeledin,Ylin)
        %% ConstantData : Global Variable for Constant Data variables
        set.initL    = initL; 
        set.unlabeled= unlabeledin;
        set.n_l  = numel(get.initL);
        set.n_u               = numel(get.unlabeled);
        set.n_q               = numel(get.query);
        set.K        = ModelInfo.Kernel;
        set.Q        = eye(size(cnstData.K,1));% or use Q. 
        set.n_o      = n_oin;
        set.n_lbn    = 0.1;
        set.n_S      = get.n_l+get.n_u;        % use all of labeled+unlabeled data
        set.n        = get.n_l+get.n_u+cnstData.n_q;    % size of alphatild ( alpha + tau)
        set.query    = get.n_l+get.n_u+1:get.n;    % indecies of cnstData.queryidx points
        set.nSDP     = get.n + 1;          % size of the SDP Matrix 
        set.nConic   = get.nSDP*get.nSDP + get.n_S;
        set.batchSize= batchSizein;
        set.KE       = [get.K,get.K(:,get.unlabeled);get.K(get.unlabeled,:),get.K(get.unlabeled,get.unlabeled)];
        set.KEn      = size(get.KE,1);
        set.KEp      = [get.KE,zeros(get.KEn,1);zeros(1,get.KEn+1)];
        set.KEvec    = reshape(get.KEp,numel(get.KEp),1); 
        set.Kuvec    = [get.KEvec;zeros(get.n_S,1)];
        set.KQinv    = inv(learningparams.lambda_o*get.K+learningparams.rho*get.Q);
        set.Yl       = Ylin;
        end
    end
end