function [query_id] = marginsampling(trainsamples, initL, samples_to_query_from, K, lambda, learningparams, alparams) 
global cnstData
   xtrain = trainsamples.X;
   ytrain = trainsamples.Y;
   
   [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);
   
   [model]                                    = svmtrainwrapper(learningparams, trainsamples, Lind,xtrain(:,Lindex), ytrain(Lindex));
   [predict_label, accuracy, decision_values] = svmpredictwrapper(model, learningparams, trainsamples, Uind, xtrain(:,Uindex), ytrain(Uindex));
   abs_dec_val                                = abs(decision_values);
   ind                                        = k_mostsmallest(abs_dec_val ,cnstData.batchSize);
   
   query_ind  = Uind(ind);
   query_id                                   = get_ID_fromind(trainsamples, query_ind);
   assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in the distributions ID in learning data');
   if ismember(query_id, initL)
       query_id
   end
end