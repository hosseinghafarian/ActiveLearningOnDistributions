function testforyalmip
lambda_o = 0.1;
soltype  = 1;
tol      = 0.001;
rho      = 1;
load('dataforyalmip','y_ICtil','y_IVtil','Stil','Z','v','x_G','operators','learningparams','optparams','cnstData');
rhoml = 1/(lambda_o*rho+1);
n_EC  = size(operators.b_EC,1);
n_EV  = size(operators.b_EV,1);
y_I   = [y_ICtil;y_IVtil];
y_EC  = sdpvar(n_EC,1);
y_EV  = sdpvar(n_EV,1);
s_I   = [operators.s_IC;operators.s_IV];
    
dualvars     = dualvar_conv(y_EC, y_EV, y_ICtil, y_IVtil, Stil, Z ,  v );
cObjective   = -dual_regwo_objective(dualvars, x_G, operators, learningparams, optparams,cnstData);
%cConstraint= [ Stil>=0,pdu>=0,v>=0,Zpartq>=0];%,v<=arho*(s_I-gprox)+y_I];%,Z(1:n_S,:)==0,Z(:,1:n_S)==0,Z(n_S+1:nSDP,n_S+1:nSDP)>=0];%,Z(nSDP,query)>=0];
Q           = cnstData.Q;
KQinv       = cnstData.KQinv;
K           = cnstData.K;
cObjective2 = -(operators.b_EC'*y_EC+operators.b_EV'*y_EV) ...
             + 1/(2*rho)*norm(operators.A_EC'*y_EC+operators.A_IC'*y_ICtil+operators.A_EV'*y_EV+operators.A_IV'*y_IVtil+Stil+Z+x_G.u/rho)^2 ...
             +1/2*(operators.B_EV'*y_EV+operators.B_IV'*y_IVtil+rho*Q*x_G.w_obeta)'*KQinv*(operators.B_EV'*y_EV+operators.B_IV'*y_IVtil+rho*Q*x_G.w_obeta)-lambda_o/2*x_G.w_obeta'*K*x_G.w_obeta ...
             +1/(2*rho)*norm(-v+y_I-x_G.st/rho)^2-v'*s_I-1/(2*rho)*norm(x_G.u,'fro')^2-1/(2*rho)*norm(x_G.st)^2;

sol2   = optimize([],cObjective2);
dobj2 = value(cObjective2);
sol   = optimize([],cObjective);
dobj  = -value(cObjective);

if sol.problem==0 
   dobj    = -value(cObjective);
   dualvars.y_EC  =  value(y_EC); 
   dualvars.y_EV  =  value(y_EV);
end
end
function [dualvars] = dualvar_conv(y_EC, y_EV, y_IC, y_IV, S, Z ,  v )
    dualvars.y_EC  = y_EC;
    dualvars.y_IC  = y_IC;
    dualvars.y_EV  = y_EV;
    dualvars.y_IV  = y_IV;
    dualvars.S     = S;
    dualvars.Z     = Z;
    dualvars.v     = v;
end
function [f_dual] = dual_regwo_objective(dualvars_k, x_0, operators, learningparams, optparams,cnstData)
    Aysdp         = operators.A_EC'*dualvars_k.y_EC + operators.A_IC'*dualvars_k.y_IC + ...
                    operators.A_EV'*dualvars_k.y_EV + operators.A_IV'*dualvars_k.y_IV;
    Bysdp         = operators.B_EV'*dualvars_k.y_EV + operators.B_IV'*dualvars_k.y_IV;
    y_I           = [dualvars_k.y_IC;dualvars_k.y_IV];
    s_I           = [operators.s_IC; operators.s_IV];
    f_dual        = ( operators.b_EC'*dualvars_k.y_EC+ operators.b_EV'*dualvars_k.y_EV)...
                     - 1/2*norm(Aysdp + dualvars_k.S + dualvars_k.Z + x_0.u)^2 ...
                     - learningparams.rhox/2*(Bysdp+cnstData.Q*x_0.w_obeta)'*cnstData.KQinv*(Bysdp+cnstData.Q*x_0.w_obeta)...
                     - 1/2*norm(dualvars_k.v + y_I- x_0.st)^2-dualvars_k.v'*s_I...
                     + 1/2*x_norm(x_0,cnstData.Q) + learningparams.lambda_o/(2*learningparams.rhox)*x_0.w_obeta'*cnstData.K*x_0.w_obeta;
end