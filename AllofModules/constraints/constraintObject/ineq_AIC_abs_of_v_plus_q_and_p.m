function [R_IC, s, size_of_s]                = ineq_AIC_abs_of_v_plus_q_and_p(c_mul_pAndw_o)
%cConstraint= [cConstraint,0<=q], where is q<=1? q<=1 is not necessary
%because of the constraint p+q <=1, all are positive. so, this constraint
%is deleted. 
% cConstraint= [cConstraint,r>=G_plus(setunlab,nap+1),r>=-G_plus(setunlab,nap+1)];
% Constraint: v <= a= 1-p-q :=> v+p+c_mul_pAndw_o*q <=1 
global cnstData
ap    = zeros(cnstData.n_S,1);
t     = 1;
for k = 1:cnstData.n_u
   ku        = cnstData.unlabeled(k);
   kq        = cnstData.extendInd(k);
   R         = sparse([ku           , cnstData.nSDP, kq               , cnstData.nSDP     ],...
                      [cnstData.nSDP, ku           , cnstData.nSDP    , kq                ],...
                      [0.5          , 0.5          , 0.5*c_mul_pAndw_o, 0.5*c_mul_pAndw_o ],...
                      cnstData.nSDP,cnstData.nSDP);
   ap(ku)    = 1; 
   R_IC(:,t) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',ap'];
   s(t,1)    = 1;
   ap(ku)    = 0;
   t = t+1; 
end
%Constraint: -v - a<=0 :=>-v+p+c_mul_pAndw_o*q<=1
for k = 1:cnstData.n_u
   ku        = cnstData.unlabeled(k);
   kq        = cnstData.extendInd(k);
   %R         = sparse([ku,cnstData.nSDP,kq,cnstData.nSDP],[cnstData.nSDP,ku,cnstData.nSDP,kq],[-0.5,-0.5,0.5,0.5],cnstData.nSDP,cnstData.nSDP);
   R         = sparse([ku           , cnstData.nSDP, kq               , cnstData.nSDP     ],...
                      [cnstData.nSDP, ku           , cnstData.nSDP    , kq                ],...
                      [-0.5         , -0.5         , 0.5*c_mul_pAndw_o, 0.5*c_mul_pAndw_o ],...
                      cnstData.nSDP,cnstData.nSDP);
   ap(ku)    = 1; 
   R_IC(:,t) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',ap'];
   s(t,1)    = 1;
   ap(ku)    = 0;
   t = t+1; 
end
size_of_s    = t-1;
% %Constraint: p+q<=1
% for k = 1:n_u
%    ku        = unlabeled(k);
%    kq        = query(k);
%    R         = sparse([kq,nSDP],[nSDP,kq],[0.5,0.5],nSDP,nSDP);
%    ap(ku)    = 1; 
%    A_IC(:,j) = [reshape(R,nSDP*nSDP,1)',ap',zeros(2*n,1)'];
%    s_IC(j,1) = 1;
%    ap(ku)    = 0;
%    j = j+1; 
% end  
end