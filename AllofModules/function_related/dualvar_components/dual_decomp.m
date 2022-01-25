function [y_EC, y_EV, y_IC, y_IV, S, Z , v]    = dual_decomp(dual_app,ind_EC, ind_EV, ind_IC, ind_IV, ind_S, ind_Z, ind_v)
    y_EC = dual_app(ind_EC);
    y_EV = dual_app(ind_EV);
    y_IC = dual_app(ind_IC);
    y_IV = dual_app(ind_IV);
    S    = dual_app(ind_S );
    Z    = dual_app(ind_Z );
    v    = dual_app(ind_v );
end