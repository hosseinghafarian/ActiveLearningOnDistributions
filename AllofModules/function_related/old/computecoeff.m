function [c_k]                         = computecoeff(lambda,c_a,alphapre)
global cnstData % KE,nSDP,n_S,query,unlabeled
nSDP  = cnstData.nSDP;
n_S   = cnstData.n_S;
%% Computing Coefficeint: c_k
% c_a   =2;
% in the following lines since g is equal to p+q, coefficeints in
% optimization is computed based on that
C_kMS     = [(alphapre*alphapre').*cnstData.KE/(2*lambda),zeros(nSDP-1,1);zeros(1,nSDP-1),0];% coefficients for G
alphatild = alphapre(1:n_S);
n_q       = numel(cnstData.extendInd);
q_ind     = [repmat(nSDP,n_q,1),cnstData.extendInd'];  % this is the indexes of q
cq        = alphatild(cnstData.unlabeled)+c_a*ones(size(cnstData.unlabeled,1),1);
% coefficients for g (which substituted by g=p+q) so, it is considered for q
% and p seperately. alphatild'*(1-g)=alphatild'*1-alphatild'*q-alphatild'*q 
C_kMq     = sparse([q_ind(:,1)',q_ind(:,2)'],[q_ind(:,2)',q_ind(:,1)'],[cq'/2,cq'/2]);
C_kM      = C_kMS + C_kMq;
cp        = alphatild+c_a*ones(n_S,1);
c_k       = [reshape(C_kM,nSDP*nSDP,1);cp];%ca;cg];%first part for C_kM, the other parts for pag: p,a,g

end
