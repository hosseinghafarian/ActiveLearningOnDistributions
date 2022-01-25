function SpectK = SpectTruncate(K, threshold)
global cnstData
     n    = size(K,1);
%     [U,D]= eig(K+ lambda *eye(n));
%     D    = diag(D);
    DS     = zeros(n,1);
    iLG    = (cnstData.D>threshold);
    DS(iLG)= 1./cnstData.D(iLG);
    SpectK = cnstData.U*diag(DS)*cnstData.U';
    