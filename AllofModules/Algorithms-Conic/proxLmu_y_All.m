function [y_ECd, y_EVd, y_ICd, y_IVd]           = proxLmu_y_All(objectivefunc, soltype, tol, maxit, ...
                                      y_ECtil, y_EVtil, y_ICtil, y_IVtil, Stil, Z , v, x_G, operators, learningparams,optparams)                             
n_EC  = size(operators.b_EC,1);
n_EV  = size(operators.b_EV,1);
b_E   = [operators.b_EC;operators.b_EV];
y_Epre= [y_ECtil;y_EVtil];
%% which method is the fastest and the best: it seems that atleast for small problems: Cholesky back and forwth solve. method 2 
if soltype == 2 %% conjugate gradient
    retType           = 1; % y_E and pcg
    LHSRHSfunc        = objectivefunc.LHSRHS;
    [f_dual, Mat, RHS] = LHSRHSfunc(retType, b_E, y_ECtil, y_EVtil, y_ICtil, y_IVtil, Stil, Z , v, x_G, operators, learningparams, optparams);
    RHSsp = sparse(RHS);
    L     = ichol(RHSsp);
    [y_E,flag,relres,iter] = pcg(Mat,RHS,tol,maxit,L,L',y_Epre);
    assert(flag==0,'pcg didnot converge in computing y_E');
    y_ECd = y_E(1:n_EC);
    y_EVd = y_E(n_EC+1:n_EC+n_EV);
elseif soltype == 3 %% cholesky factorization
    retType           =  2; % y_E and cholesky
    LHSRHSfunc        = objectivefunc.LHSRHS;
    [f_dual, Mat, RHS] = LHSRHSfunc(retType, b_E, y_ECtil, y_EVtil, y_ICtil, y_IVtil, Stil, Z , v, x_G, operators, learningparams, optparams);
    y_E1         = Mat'  \ RHS;
    y_E          = Mat   \ y_E1;
    y_ECd        = y_E(1:n_EC);
    y_EVd        = y_E(n_EC+1:n_EC+n_EV);
elseif soltype == 1 
    y_EC         = sdpvar(n_EC,1);
    y_EV         = sdpvar(n_EV,1);
    y_IC         = sdpvar(operators.n_AIC,1);
    y_IV         = sdpvar(operators.n_AIV,1);
    dualobjfunc  = objectivefunc.dual;
    dualvars     = dualvar_conv(y_EC, y_EV, y_IC, y_IV, Stil, Z ,  v );
    cObjective   = -dualobjfunc(dualvars, x_G, operators, learningparams, optparams);
    sol          = optimize([],cObjective);
    if sol.problem==0 
       obj_val   = -value(cObjective);
       y_ECd     =  value(y_EC); 
       y_EVd     =  value(y_EV);
       y_ICd     =  value(y_IC);
       y_IVd     =  value(y_IV);
    else 
        assert(true,'Error cannot solve problem for y_E in routine proxLmu_y_E');
    end
end    
end