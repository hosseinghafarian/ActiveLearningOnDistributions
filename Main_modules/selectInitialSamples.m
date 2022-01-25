function  [ind] = selectInitialSamples(al_profile, trainsamples)
initType = al_profile.init_type;

%% Initial Samples Selection : How to initialy select random samples? 
% 1: Select Two random samples
% 2: Select Two random samples from different classes
% 3: All of samples (Passiver learning)
% 4: select sample by user.
%% Select initial samples 
ind      = initialSamplesSelected(al_profile, initType, trainsamples);
end