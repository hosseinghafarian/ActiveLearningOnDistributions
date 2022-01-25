function [x_k ] = x_conv_from_dual_project_st(y_EC,y_EV,y_IC,y_IV,S,Z,v, x_0, operators)
%% This function makes x from dual variables. 
%  It's result is not correct if lagrange variables doesnot obey certain
%  conditions which makes x_k components satisfy their constraints. 
%  This is only done for x_k.st
global cnstData

    x_k.u         = x_0.u + operators.A'*[y_EC;y_EV;y_IC;y_IV] ...
                          + S + Z;
    x_k.w_obeta   = x_0.w_obeta + ...
                    cnstData.Hinv*(operators.B_EV'*y_EV + operators.B_IV'*y_IV);
    x_k.st        = min(x_0.st - (v+[y_IC;y_IV]), [operators.s_IC;operators.s_IV]);
end