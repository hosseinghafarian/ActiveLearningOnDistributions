function [dualvars, obj_val_dual, sol_problem_dual] = solve_dual(objectivefunc_dual,x_0,operators,learningparams,optparams)
global cnstData
    n_EC          = operators.n_AEC;
    n_IC          = operators.n_AIC;
    n_EV          = operators.n_AEV;
    n_IV          = operators.n_AIV;
    
    y_EC          = sdpvar(n_EC,1); 
    y_IC          = sdpvar(n_IC,1);
    y_EV          = sdpvar(n_EV,1); 
    y_IV          = sdpvar(n_IV,1);
    S_mat         = sdpvar(cnstData.nSDP,cnstData.nSDP);
    p_NN_dual     = sdpvar(cnstData.n_S,1);
    q_NN_dual     = sdpvar(cnstData.n_u,1);
    v             = sdpvar(n_IC+n_IV,1);

    last_col_dual = [zeros(cnstData.n_S,1);q_NN_dual];
    Z_NN_Mat      = [zeros(cnstData.nSDP-1,cnstData.nSDP-1),last_col_dual;last_col_dual',0];
    Z_NN          = [reshape(Z_NN_Mat,cnstData.nSDP*cnstData.nSDP,1);zeros(cnstData.n_S,1)]; 
    S             = [reshape(S_mat,cnstData.nSDP*cnstData.nSDP,1);p_NN_dual];

    Aysdp         = operators.A_EC'*y_EC+ operators.A_IC'*y_IC+ operators.A_EV'*y_EV+ operators.A_IV'*y_IV;
    Bysdp         = operators.B_EV'*y_EV+ operators.B_IV'*y_IV;
    y_I           = [y_IC;y_IV];
    s_I           = [operators.s_IC; operators.s_IV];

    dcConstraint  = [ S_mat>=0, p_NN_dual>=0, v>=0, q_NN_dual>=0];
    %% Warning: The following function must be check and be substitute the next lines for dcObjective.
    dualvars      = dualvar_conv(y_EC, y_EV, y_IC, y_IV, S, Z_NN ,  v );
    
    dcObjective   = -objectivefunc_dual(dualvars, x_0,operators,learningparams,optparams);
    Options       = sdpsettings('dualize',0,'verbose',0);
    sol           = optimize(dcConstraint,dcObjective,Options);
    obj_val_dual  = -value(dcObjective);
    sol_problem_dual = sol.problem;  
    
    dualvars      = dualvar_conv(value(y_EC), value(y_EV), value(y_IC), value(y_IV), value(S), value(Z_NN) ,  value(v) );
end
