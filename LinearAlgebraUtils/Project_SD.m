function XP=Project_SD(X)
eps = 10^-6;
[V,D]=eig((X+X')/2);
D_p = D;
for i=1:size(X,1)
    if D_p(i,i)<eps
        D_p(i,i)=0;
    end
end
XP = V*D_p*V';
XP = (XP + XP')/2;
end