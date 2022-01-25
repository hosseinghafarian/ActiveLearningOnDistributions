function [fout,gout]                   = f_lG_x_alpha(alphav)
    global KG;global h_of_x;global alphapref;global rhop;

    fout = -alphav'*h_of_x + 1/2* alphav'*KG*alphav + rhop/(2)*norm(alphav-alphapref)^2;
    gout = -        h_of_x +              KG*alphav + rhop*(alphav-alphapref);
end