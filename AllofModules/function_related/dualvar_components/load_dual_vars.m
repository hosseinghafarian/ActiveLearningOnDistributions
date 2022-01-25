function [y_ECtil,y_EVtil,y_ICtil,y_IVtil,Stil,Ztil,vtil] = load_dual_vars(dualvarsPre)
    y_ECtil  = dualvarsPre.y_EC ;
    y_EVtil  = dualvarsPre.y_EV ;
    y_ICtil  = dualvarsPre.y_IC ;
    y_IVtil  = dualvarsPre.y_IV ;
    Stil     = dualvarsPre.S    ; 
    Ztil     = dualvarsPre.Z    ;
    vtil     = dualvarsPre.v    ;
end