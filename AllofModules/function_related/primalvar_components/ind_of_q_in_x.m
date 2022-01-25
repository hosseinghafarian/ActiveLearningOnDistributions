function [ind ]                        = ind_of_q_in_x()
global cnstData 
       sub_1     = repmat(cnstData.nSDP, cnstData.n_q,1);
       ind       = sub2ind([cnstData.nSDP,cnstData.nSDP], sub_1,cnstData.extendInd');
end