function [x_k ] = x_conv_from_dual(dualvars, x_0, operators)
% %% This function makes x from dual variables. 
%  It's result is not correct if lagrange variables doesnot obey certain
%  conditions which makes x_k components satisfy their constraints. 
%  As an important note, if v and y_I doesnot obey these conditions then 
%  x_k.st doesnot satisfy the following constraints: x_k.st <=
%  [operators.s_IC;operators.S_IV];
    [x_k ] = x_conv_from_dual_noproject(dualvars.y_EC,dualvars.y_EV,dualvars.y_IC,dualvars.y_IV,...
                                        dualvars.S   ,dualvars.Z   ,dualvars.v   , x_0, operators);
end