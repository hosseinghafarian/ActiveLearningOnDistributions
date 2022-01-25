function setup_paths(mainfolderpath)
addpath(genpath(mainfolderpath));
%     addpath(genpath(strcat(mainfolderpath,'\libsvm-3.21')));  %for libsvm
%     addpath(genpath(strcat(mainfolderpath,'\AllofModules')));
%     addpath(genpath(strcat(mainfolderpath,'\AllofModules\function_related')));  %for function .m files
%     addpath(genpath(strcat(mainfolderpath,'\AllofModules\constraints')));  %for function .m files
%     addpath(genpath(strcat(mainfolderpath,'\AllofModules\cone_related')));  %for function .m files
%     addpath(genpath(strcat(mainfolderpath,'\AllofModules\convexSDPformulation')));  %for function .m files
%     addpath(genpath(strcat(mainfolderpath,'\AllofModules\global_data')));  %for function .m files
%     addpath(genpath(strcat(mainfolderpath,'\AllofModules\Algorithms-SaddlePoint')));  %for function .m files
%     addpath(genpath(strcat(mainfolderpath,'\AllofModules\Algorithms-Quadratic')));  %for function .m files
%     addpath(genpath(strcat(mainfolderpath,'\AllofModules\Algorithms-Conic')));  %for function .m files
%     addpath(genpath(strcat(mainfolderpath,'\AllofModules\settings')));  %for function .m files
%     addpath(genpath(strcat(mainfolderpath,'\active_learning_strategies')));
%     addpath(genpath(strcat(mainfolderpath,'\ComparedAgainst')));         %codes for other active learning methods
%    addpath(genpath(strcat(mainfolderpath,'\Experiments')));             % experiments files
%     addpath(genpath(strcat(mainfolderpath,'\fig_files_experiments')));   % experiments fig files
%     addpath(genpath(strcat(mainfolderpath,'\learning_methods')));        % other classifier methods .
%     addpath(genpath(strcat(mainfolderpath,'\Libs_ext')));                % external libraries like libsvm
%     addpath(genpath(strcat(mainfolderpath,'\Mylib')));                % external libraries like libsvm
%     addpath(genpath(strcat(mainfolderpath,'\Main_modules')));            % Mainmodules
%     addpath(genpath(strcat(mainfolderpath,'\Solve_subproblems_withYALMIP')));  % codes written to solve a problem with YALMI
%     addpath(genpath(strcat(mainfolderpath,'\utils')));                   % utility functions
%     addpath(genpath(strcat(mainfolderpath,'\Datasets')));                   % utility functions
%     addpath(genpath(strcat(mainfolderpath,'\DISTDatasets')));                   % utility functions DISTributional
%     addpath(genpath(strcat(mainfolderpath,'\DISTributional')));                   % utility functions 
%     addpath(genpath(strcat(mainfolderpath,'\KERNELS')));                   % utility functions 
%     addpath(genpath(strcat(mainfolderpath,'\unlabeled_subsampling')));                   % selecting a subset of unlabeled instances for sampling
%     addpath(genpath(strcat(mainfolderpath,'\DataSet_subsampling')));                   % selecting a subset of unlabeled instances for sampling
%     addpath(genpath(strcat(mainfolderpath,'\noise_label_mapping')));                   % selecting a subset of unlabeled instances for sampling
%     addpath(genpath(strcat(mainfolderpath,'\OptimizationSolves\FASTA')));                   % selecting a subset of unlabeled instances for sampling
%     addpath(strcat(mainfolderpath,'\nonconvex'));                   % selecting a subset of unlabeled instances for sampling
    addpath(genpath('E:\Code\Code\Datasets')); 
    %addpath(genpath('I:\MyPapers971112\Papers\ARL\Datasets')); 
end