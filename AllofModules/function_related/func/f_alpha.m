function [fout,gout]                   = f_alpha(alphav)
global MC;global lambdaf;global alphapref;global rhop;global g_alpha;
%MC = (K.* Xpre(1:n,1:n))/lambdaf;
fout = -alphav'*(1-g_alpha)+1/2*alphav'*MC*alphav+1/(2*rhop)*norm(alphav-alphapref)^2;
gout = -(1-g_alpha) + MC*alphav+1/rhop*(alphav-alphapref);
%fout = -alphav'*(1-g_alpha)+alphav'*MC*alphav+1/(2*rhop)*norm(alphav-alphapref)^2;
%gout = -1 + MC*alphav+1/rhop*(alphav-alphapref);
end