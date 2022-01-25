function [y_ICd, y_IVd]           = proxLmu_y_I(objectivefunc,soltype,tol,maxit,...
                                   y_ECtil, y_EVtil, y_ICtil, y_IVtil, Stil, Z , v, x_G, operators,learningparams,optparams)
%% which method is the fastest and the best: it seems that atleast for small problems: Cholesky back and forwth solve. method 2 
y_Ipre                = [y_ICtil;y_IVtil];
b_E                   = [operators.b_EC;operators.b_EV];
n_IC                  = size(operators.s_IC,1);
n_IV                  = size(operators.s_IV,1);
if soltype == 2 %% conjugate gradient
    retType            = 3; % y_I and pcg
    LHSRHSfunc         = objectivefunc.LHSRHS;
    [Mat, RHS] = LHSRHSfunc(retType, b_E, y_ECtil, y_EVtil, y_ICtil, y_IVtil, Stil, Z , v, x_G, operators, learningparams, optparams);
    RHSsp              = sparse(Mat);
    L                  = ichol(RHSsp); %L                  = eye(size(Mat)); 
    [y_Id,flag,relres,iter] = pcg(Mat,RHS,tol,maxit,L,L',y_Ipre);
    assert(flag==0,'pcg didnot converge in computing y_E')
    y_ICd              = y_Id(1:n_IC);
    y_IVd              = y_Id(n_IC+1:n_IC+n_IV);
elseif soltype == 3 %% cholesky factorization
    retType            =  4; % y_I and cholesky
    LHSRHSfunc         = objectivefunc.LHSRHS;
    [Mat, RHS] = LHSRHSfunc(retType, b_E, y_ECtil, y_EVtil, y_ICtil, y_IVtil, Stil, Z , v, x_G, operators, learningparams, optparams);
    y_I1               = Mat'\RHS;
    y_I                = Mat \y_I1;
    y_ICd              = y_I(1:n_IC);
    y_IVd              = y_I(n_IC+1:n_IC+n_IV);
    %[f_dual, g_y_E, g_y_I] = dual_regwo_objective_split_grad(b_E,y_ECtil, y_EVtil, y_ICd, y_IVd, Stil, Z , v, x_G, operators, learningparams, optparams);
elseif soltype == 1 
    y_IC               = sdpvar(operators.n_AIC,1);
    y_IV               = sdpvar(operators.n_AIV,1);    
    dualvars           = dualvar_conv(y_ECtil, y_EVtil, y_IC, y_IV, Stil, Z ,  v );
    dualobjfunc        = objectivefunc.dual;
    cObjective         = -dualobjfunc(dualvars, x_G, operators, learningparams, optparams);
    sol                = optimize([],cObjective);
    if sol.problem    == 0 
        obj_val        = -value(cObjective);
        y_ICd          = value(y_IC); 
        y_IVd          = value(y_IV);
    else
        assert(true,'Could not optimize for value of y_I in function proxLmu_y_I');
    end
end
end
