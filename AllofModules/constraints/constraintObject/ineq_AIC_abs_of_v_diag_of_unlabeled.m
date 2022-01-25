function [R_IC, s, size_of_s]                = ineq_AIC_abs_of_v_diag_of_unlabeled()
    % In the following r==G_plus(setunlab,setunlab)
    % cConstraint= [cConstraint,r>=G_plus(setunlab,nap+1),r>=-G_plus(setunlab,nap+1)];
    % Constraint: a= 1-p-q>= v :=> v-a<=0 :=> G_plus(setunlab, nap+1)-G_plus(setunlab,setunlab)<=0 
global cnstData
    ap        = zeros(cnstData.n_S,1);
    t = 1;
    for k = 1:cnstData.n_u
       ku        = cnstData.unlabeled(k);
       R         = sparse([ku,cnstData.nSDP,ku],[cnstData.nSDP,ku,ku],[0.5,0.5,-1],cnstData.nSDP,cnstData.nSDP);
       R_IC(:,t) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',ap'];
       s(t,1)    = 0;
       t = t+1; 
    end
    % Constraint: a= 1-p-q>= -v :=> -v-a<=0 :=> -G_plus(setunlab, nap+1)-G_plus(setunlab,setunlab)<=0 
    for k = 1:cnstData.n_u
       ku        = cnstData.unlabeled(k);
       R         = sparse([ku,cnstData.nSDP,ku],[cnstData.nSDP,ku,ku],[-0.5,-0.5,-1],cnstData.nSDP,cnstData.nSDP);
       R_IC(:,t) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',ap'];
       s(t,1)    = 0;
       t = t+1; 
    end
    size_of_s    = t-1;
end