function SpectK = L2BoostingSpectral(K, lambda, t)
global cnstData
     n    = size(K,1);
%     [U,D]= eig(K+ lambda *eye(n));
%     D    = diag(D);
    eta  = 1/(max(cnstData.D)+lambda);
    oned = ones(n,1); 
    RT   = oned-eta*(cnstData.D+lambda*ones(size(cnstData.D)));
    RTT  = oned;
    DS   = oned;
    for i=1:t
        RTT = RTT.*RT;
        DS  = DS + RTT;
    end
    DS     = eta* DS;
    SpectK = cnstData.U*diag(DS)*cnstData.U';
end