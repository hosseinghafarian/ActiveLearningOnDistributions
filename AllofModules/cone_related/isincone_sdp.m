function is_sdp  = isincone_sdp(z,n)
tolneg = 10^-3;
is_sdp = false;
if n==0
    return;
elseif n==1
    if z> 0
        is_sdp = true;
    end
    return
end
[~,S] = eig(z);
S = diag(S);
idx = find(S<-tolneg);
if numel(idx) ==0 
    is_sdp = true;
end    
end