function Plotfilename   = get_filename(datasetName, gamma, gamma_o, lambda, lambda_o)
        Plotfilename  = strcat('Plot_DS=',datasetName);
        Plotfilename  = sprintf('_%s=%s', Plotfilename, datestr(datetime(),'yyyy_mmmm_dd_HH_MM'));
        Plotfilename  = sprintf('%s=%4e', Plotfilename, gamma);  
        Plotfilename  = sprintf('%s-%s',  Plotfilename, '_SO'); %SO for gamma_o
        Plotfilename  = sprintf('%s=%4e', Plotfilename, gamma_o); 
        Plotfilename  = sprintf('%s-%s',  Plotfilename, '_LM');  %LM for lambda
        Plotfilename  = sprintf('%s=%4e', Plotfilename, lambda); 
        Plotfilename  = sprintf('%s-%s',  Plotfilename, '_LO');  %LO for lambda_o
        Plotfilename  = sprintf('%s=%4e', Plotfilename, lambda_o);
%        Plotfilename  = strcat(Plotfilename, '.mat');
end