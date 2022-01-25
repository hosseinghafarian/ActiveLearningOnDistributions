function SpectK = TiknonovSpectral(K, lambda, t)
global cnstData
    n    = size(K,1);
    DPL  = (cnstData.D+lambda*ones(n,1)).^t;
    DS   = (DPL-lambda^t*ones(n,1))./(DPL.*cnstData.D);
    SpectK = cnstData.U*diag(DS)*cnstData.U';
end