function [x_ans] = x_sum( x_k, x_a)
 x_ans.u       = x_k.u+x_a.u;
 x_ans.w_obeta = x_k.w_obeta+x_a.w_obeta;
 x_ans.st      = x_k.st+ x_k.st;
end