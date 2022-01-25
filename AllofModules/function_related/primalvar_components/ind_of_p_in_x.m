function [ind ]  = ind_of_p_in_x()
global cnstData 
       nSDP = cnstData.nSDP;
       n_S  = cnstData.n_S;
       i_p_s= nSDP*nSDP;
       i_p_e= i_p_s + n_S;
       ind  = i_p_s+1:i_p_e;
end