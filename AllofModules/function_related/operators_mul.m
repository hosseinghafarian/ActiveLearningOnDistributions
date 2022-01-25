function [ Ay ] = operators_mul(y_EC,y_IC,y_EV,y_IV,operators)
% Instead use the following: 
%    A        = [operators.A_EC;operators.A_EV;operators.A_IC;operators.A_IV];
%    Ay       = A'*[y_EC;y_EV;y_IC;y_IV];

   Ay          = operators.A_EC'*y_EC + operators.A_IC'*y_IC + operators.A_EV'*y_EV + operators.A_IV'*y_IV;
end