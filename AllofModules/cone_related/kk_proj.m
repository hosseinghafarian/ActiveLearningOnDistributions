function px                            = kk_proj(x,kku,kkl)
ndim=length(x);
px=zeros(ndim,1);
px=min(kku,x); 
px=max(kkl,px);
end
