function w = ProjectOntoSimplex(x, b)
% PROJECTONTOSIMPLEX Projects point onto simplex of specified radius.
% 
% w = ProjectOntoSimplex(x, b) returns the vector w which is the solution
%   to the following constrained minimization problem:
% 
%    min   ||w - x||_2
%    s.t.  sum(w) <= b, w >= 0.
% 
%   That is, performs Euclidean projection of x to the positive simplex of
%   radius b.
% 
% Author: John Duchi (jduchi@cs.berkeley.edu)

if (b < 0)
  error('Radius of simplex is negative: %2.3f\n', b);
end
x = (x > 0) .* x;
u = sort(x,'descend');
sv = cumsum(u);
rho = find(u > (sv - b) ./ (1:length(u))', 1, 'last');
theta = max(0, (sv(rho) - b) / rho);
w = max(x - theta, 0);
end
function w = myProjectOntoSimplex(x, b)
n = numel(x);
vlow = 0;
objlow = objectivefun(x,vlow,n);
vhigh= 10;
objhigh= objectivefun(x,vhigh,n);
while vlow+0.0001<vhigh
    mid   =(vlow+vhigh)/2;
    objmid=objectivefun(x,mid,n);
    if objmid >objhigh
        vhigh = mid;
        objhigh = objmid;
    elseif  objmid <= objhigh
        vlow = mid;
        objlow = objmid;
    end
end
w=max(x-mid*1,0);
end
function [obj]= objectivefun(x,v,n)
obj = 1/2* norm(min(x-v*1,0),2)^2+v*(sum(x)-1)-n*v^2;

end
function amain
x = [0.9,0.1,0.1]';
tic
w= myProjectOntoSimplex(x,1);
toc
sum(w)
tic
w1=JProjectOntoSimplex(x, 1);
toc
sum(w1)
end