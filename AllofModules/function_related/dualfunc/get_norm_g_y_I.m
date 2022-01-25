function [n_g_y_I] = get_norm_g_y_I(b_E,y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams)
global cnstData 
   y_I      = [y_IC;y_IV];
   Ay       = operators.A'*[y_EC;y_EV;y_IC;y_IV];
   By       = operators.B'*[y_EC;y_EV;y_IC;y_IV];
   SumAySZu = Ay + S + Z + x_0.u;
   n_g_y_I    = norm(-operators.A_I*(SumAySZu)-operators.B_I*(cnstData.Hinv*By+x_0.w_obeta)...
                     -(v+y_I-x_0.st));
end