function setup_paths(mainfolderpath)
    addpath(genpath(strcat(mainfolderpath,'\libsvm-3.21')));  %for libsvm
    addpath(genpath(strcat(mainfolderpath,'\AllofModules')));
    addpath(genpath(strcat(mainfolderpath,'\AllofModules\function_related')));  %for function .m files
    addpath(genpath(strcat(mainfolderpath,'\AllofModules\constraints')));  %for function .m files
    addpath(genpath(strcat(mainfolderpath,'\AllofModules\cone_related')));  %for function .m files
    addpath(genpath(strcat(mainfolderpath,'\AllofModules\convexSDPformulation')));  %for function .m files
    addpath(genpath(strcat(mainfolderpath,'\AllofModules\global_data')));  %for function .m files
    addpath(genpath(strcat(mainfolderpath,'\AllofModules\Algorithms-SaddlePoint')));  %for function .m files
    addpath(genpath(strcat(mainfolderpath,'\AllofModules\Algorithms-Quadratic')));  %for function .m files
    addpath(genpath(strcat(mainfolderpath,'\AllofModules\Algorithms-Conic')));  %for function .m files
    addpath(genpath(strcat(mainfolderpath,'\AllofModules\settings')));  %for function .m files
    addpath(genpath(strcat(mainfolderpath,'..\Datasets'))); 
end