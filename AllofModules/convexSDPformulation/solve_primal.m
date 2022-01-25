function [v_k_x, obj_val, sol_problem]= solve_primal(objectivefunc_primal,x_0,operators,learningparams,optparams)
global cnstData;

    n_IC           = operators.n_AIC;
    n_IV           = operators.n_AIV;
    X              = sdpvar(cnstData.nSDP,cnstData.nSDP);
    p              = sdpvar(cnstData.n_S,1);
    w_obeta        = sdpvar(cnstData.n_S,1);
    Xr             = [reshape(X,cnstData.nSDP*cnstData.nSDP,1);p;];
    s1             = sdpvar(n_IC,1);
    s2             = sdpvar(n_IV,1);
    s              = [s1;s2];
    v_k            = x_conv(X,p,w_obeta,s);
    [pcConstraint] = primal_constraints(v_k, operators);
    pcObjective    = objectivefunc_primal(v_k, x_0, operators, learningparams, optparams);
    Options        = sdpsettings('dualize',0,'verbose',0);
    sol            = optimize ( pcConstraint, pcObjective,Options);
    obj_val        = value(pcObjective);
    sol_problem    = sol.problem;
    v_k_x.u        = value(v_k.u);
    v_k_x.w_obeta  = value(v_k.w_obeta);
    v_k_x.st       = value(v_k.st);
end