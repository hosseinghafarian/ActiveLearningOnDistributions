function [inner_ans ] = x_inner(x_k, x_a)
   inner_ans = x_k.u'*x_a.u + x_k.w_obeta'*x_a.w_obeta + x_k.st'*x_k.st;
end