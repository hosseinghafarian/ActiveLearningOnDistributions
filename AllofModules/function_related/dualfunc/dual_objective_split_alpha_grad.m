function [f_dual, g_y_E, g_y_I, g_S, g_Z, g_v, g_dual_alpha] = dual_objective_split_alpha_grad(b_E,alpha_k, y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams)
global cnstData 
   y_I      = [y_IC;y_IV];
   n_S      = cnstData.n_S;
   napp     = cnstData.nappend;
   s_I      = [operators.s_IC;operators.s_IV];
   Ay       = operators.A'*[y_EC;y_EV;y_IC;y_IV];
   By       = operators.B'*[y_EC;y_EV;y_IC;y_IV];
   SumAySZu = Ay + S + Z + x_0.u;
   [f_dual] = dual_objective_split(y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams);
   x_prime  = x_conv_u(SumAySZu,zeros(cnstData.n_S,1),zeros(size(y_I)));
   g_y_E    = b_E - operators.A_E*(SumAySZu) ...
                  - operators.B_E*(cnstData.Qinv*By +x_0.w_obeta);
   g_y_I    = -operators.A_I*(SumAySZu)-operators.B_I*(cnstData.Qinv*By+x_0.w_obeta)...
              -(v+y_I-x_0.st);
   g_S      = -(SumAySZu);
   g_Z      = -(SumAySZu);
   g_v      = - s_I - (v + y_I-x_0.st);
   %gamma = 1
   g_dual_alpha = 2*l_of_x(x_prime)-2/learningparams.lambda *(cnstData.KE.*G_of_x(x_prime))*alpha_k ...
                  -2*[alpha_k(1:n_S)+learningparams.ca;zeros(napp,1)] ...
                  -1/(learningparams.lambda)^2*(cnstData.KE.*cnstData.KE.*(alpha_k*alpha_k'))*alpha_k;
end