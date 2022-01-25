function [noisy_ind] = get_noisy_sample_ind(pv, n, lnoiseper, onoiseper, separate)
global cnstData
if ~separate
   if cnstData.label_outlier_seperate_deal
      number_of_noisyinstances = (lnoiseper+onoiseper)*n/100;
   else
      number_of_noisyinstances = cnstData.n_o; 
   end
   noisy_ind   = k_mostlargest(pv(1:n),number_of_noisyinstances);
else
   if cnstData.label_outlier_seperate_deal
      number_of_noisyinstances_labeled   = floor((lnoiseper)*cnstData.n_l/100);
      number_of_noisyinstances_unlabeled = floor((onoiseper)*cnstData.n_u/100);
      noisy_labeled_k                    = k_mostlargest(pv(cnstData.initL(cnstData.initLnozero)),number_of_noisyinstances_labeled);
      noisy_unlabeled_k                  = k_mostlargest(pv(cnstData.unlabeled),number_of_noisyinstances_unlabeled);
      noisy_ind = [cnstData.initL(noisy_labeled_k)',cnstData.unlabeled(noisy_unlabeled_k)];
   else
      number_of_noisyinstances = cnstData.n_o; 
      noisy_ind   = k_mostlargest(pv(1:n),number_of_noisyinstances);
   end
end
end
