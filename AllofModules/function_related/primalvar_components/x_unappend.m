function x_k     = x_unappend(x_app, n_u, n_w_obeta, n_st)
     x_k.u       = x_app(1:n_u);
     x_k.w_obeta = x_app(n_u+1:n_u+n_w_obeta);
     x_k.st      = x_app(n_u+n_w_obeta+1:n_u+n_w_obeta+n_st);
end