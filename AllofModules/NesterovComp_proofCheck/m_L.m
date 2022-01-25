function [m_val, f_y, grad_f_y] = m_L(y,x,x_0,L,f_func,Psi_func,dist_func, operators,psi_strcvx)
    [f_y, grad_f_y] = f_func(y, operators);
    ginnerx         = grad_f_y.u'*(x.u-y.u) + grad_f_y.w_obeta'*(x.w_obeta-y.w_obeta) + grad_f_y.st'*(x.st-y.st);
    m_val           = f_y + ginnerx + L/2*dist_func(x,y)+ Psi_func(x,x_0,operators, psi_strcvx);
end