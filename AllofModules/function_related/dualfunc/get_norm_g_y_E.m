function [n_g_y_E] = get_norm_g_y_E(b_E,y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams)
global cnstData 
   Ay       = operators.A'*[y_EC;y_EV;y_IC;y_IV];
   
   By       = operators.B'*[y_EC;y_EV;y_IC;y_IV];
   SumAySZu = Ay + S + Z + x_0.u;
   n_g_y_E    = norm(b_E - operators.A_E*(SumAySZu)- operators.B_E*(cnstData.Hinv*By +x_0.w_obeta));
end