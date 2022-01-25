function [x_ans ] = x_cmul(x_k, scal)
    x_ans.u       = x_k.u      *scal;
    x_ans.w_obeta = x_k.w_obeta*scal;
    x_ans.st      = x_k.st     *scal;
end