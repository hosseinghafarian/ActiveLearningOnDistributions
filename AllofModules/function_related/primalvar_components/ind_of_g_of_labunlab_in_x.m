function [ind ]  = ind_of_g_of_labunlab_in_x()
global cnstData 
       sub_1     = repmat(cnstData.nSDP, cnstData.n_S,1);
       ind       = sub2ind([cnstData.nSDP,cnstData.nSDP], sub_1,(1:cnstData.n_S)');
end