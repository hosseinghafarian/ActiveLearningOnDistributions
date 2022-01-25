function [f_dual, g_y_E, g_y_I, g_S, g_Z, g_v] = dual_objective_split_grad(b_E,y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams)
global cnstData 
   y_I      = [y_IC;y_IV];
   s_I      = [operators.s_IC;operators.s_IV];
   A        = [operators.A_EC;operators.A_EV;operators.A_IC;operators.A_IV];
   Ay       = A'*[y_EC;y_EV;y_IC;y_IV];
   B        = [operators.B_E;operators.B_I];
   By       = B'*[y_EC;y_EV;y_IC;y_IV];
   SumAySZu = Ay + S + Z + x_0.u;
   [f_dual] = dual_objective_split(y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams);
   g_y_E    = b_E - operators.A_E*(SumAySZu) ...
                  - operators.B_E*(cnstData.Qinv*By +x_0.w_obeta);
   g_y_I    = -operators.A_I*(SumAySZu)-operators.B_I*(cnstData.Qinv*By+x_0.w_obeta)...
              -(v+y_I-x_0.st);
   g_S      = -(SumAySZu);
   g_Z      = -(SumAySZu);
   g_v      = - s_I - (v + y_I-x_0.st);
end