function [data]=loaddataset(datasetId, profile, recompute)
%% Load datasets based on Id
% the following list of functions, subsamples from dataset, i.e., it
% selects a subset of samples for all of the subsequent
% operations(including train, activelearning queries, test) 
% DS is abbreviation for dataset 

subdataset_method                   = profile.DSsubsampling_id;
DS_subsampling_list                 = {@DS_all_sampling, @DS_random_sampling, @DS_random_sampling};
DS_all_sampling_settings.percent    = 100;
DS_random_sampling_settings.percent = profile.DSsubsampling_per;
DS_crossvalidation_settings.percent = profile.CVsubsampling_per;
DS_settings                         = {DS_all_sampling_settings, DS_random_sampling_settings, DS_crossvalidation_settings}; 
% the following list of functions, selects a subset of already selected
% samples from dataset for querying, i.e. it only selects a subset of instances from which we query.  
UN_subsampling_method               = profile.UNsubsampling_id;
UN_labeled_subset_query_list        = {@queryall_instances, @queryrandom_instances};
UN_all_sampling_settings.percent    = 100;
UN_random_sampling_settings.percent = profile.UNsubsampling_per;
UN_settings                         = {UN_all_sampling_settings, UN_random_sampling_settings};
% the following list of functions, selects a subset of already selected
% instances for training set when the training set is large
TR_subdataset_method                = profile.TRsubsampling_id;
TR_subsampling_list                 = {@TS_all_sampling, @TS_random_sampling};
TR_sampling_settings.percent        = 100;
TR_random_sampling_settings.percent = profile.TRsubsampling_per;
TR_random_sampling_settings.method_to_adjust_size = profile.TRadjustsizemethod_id; % 1 :use percentage, 2: make it no more than if_larger_than 
TR_random_sampling_settings.sample_if_larger_than = profile.TRsample_larger_than;
TR_settings                         = {TR_sampling_settings, TR_random_sampling_settings}; 
% the following list of functions, maps classes of data to our desired
% classes, which are often {-1,1}, it also, informs us of any noisy
% instances (if it comes from other multiclass labeled data, which we
% converted them to noisy data) 
LB_mapping_method                   = profile.LBmaping_id;
LB_mapfunc_list                     = {@LB_nonoisy,...                 %no outlier and no label noise addition 
                                       @LB_other_labeles_noisy,...     %outlier and label_noise_percent label noise addition 
                                       @LB_most_two_labels_binary,...  %outlier and label_noise_percent label noise addition 
                                       @LB_balanced_labeled,...        %outlier and label_noise_percent label noise addition 
                                       @LB_balanced_labeled,...        %outlier and label_noise_percent label noise addition 
                                       };
LB_mapsetting_nonoisy.class_map     = [1,1;2,-1];
LB_mapsetting_otherlabelsnoisy.class_map = [1,1;2,-1];
LB_mapsetting_otherlabelsnoisy.label_noise_percent = profile.labelnoisepercent;
LB_mapsetting_most_two_labels_binary.map_to_labels = [1,-1];
LB_mapsetting_most_two_labels_binary.label_noise_percent = profile.labelnoisepercent;
LB_mapsetting_most_two_labels_binary.down_sample_others_percent = profile.outlierpercent;%this is outlier percent

LB_balanced_labeled_settings = LB_mapsetting_most_two_labels_binary;
LB_balanced_labeled_settings.unbalanced_percent_tolerance = profile.unbalance_LB_tolerance;
LB_balanced_labeled_settings.label_noise_percent = profile.labelnoisepercent;    
LB_settings_list                    = {LB_mapsetting_nonoisy,                LB_mapsetting_otherlabelsnoisy, ...
                                       LB_mapsetting_most_two_labels_binary, LB_balanced_labeled_settings};

global cnstDefs
datasetmain= cnstDefs.dataset_path;
gamma_o_ratio  = 1;
gamma_ratio    = 1;
lambda_o_ratio = 10;
kernel_func    = @myrbf_kernel;
switch(datasetId)
    case {cnstDefs.SYNTH4OUTLIERORDERED,...%                = 30;
        cnstDefs.SYNTHSYNTH4OUTLIER_CORRECTLYLABELED,...% = 31;
        cnstDefs.SYNTHWITHOUTOUTLIERORDERED,...%          = 32;
        cnstDefs.SYNTHLINEARSVM,...%                      = 33;
        cnstDefs.SYNTHMORECOMPLEXTESTSIMPLEFUNC,...%      = 34;
        cnstDefs.SYNTHORDEREDOUTLIERLARGER,...%           = 35;
        cnstDefs.SYNTHORDEREDLITTLE_WITHOUTOUTLIER,...%   = 36;
        cnstDefs.SYNTHORDERED_DENSE_TWOOUTLIERS,...%      = 37;
        cnstDefs.SYNTHORDERED_DENSE_ONEOUTLIERS,...%      = 38;
        cnstDefs.SYNTHONEOUTLIER_ANOTHER,...%             = 39;
        cnstDefs.SYNTH2OUTLIER_FARAWAY,...%               = 40;
        cnstDefs.SYNTH4OUTLIER_INTHESAMEDIRECTION,...%    = 41;
        cnstDefs.SYNTH4OUTLIER_TWOINBOUNDRY,...%          = 42;
        cnstDefs.SYNTH4OUTLIER_TWOINBOUNDRY_MORELARGER,...% = 43;
        cnstDefs.SYNTH6OUTLIER_TWOINBOUNDRY_MORELARGER,...% = 44;
        cnstDefs.SYNTH1OUTLIER_INBOUNDRYSMALL,...%          = 45;
        cnstDefs.SYNTH1OUTLIER_INBOUNDRYSMALL_FURTHER,...%  = 46;
        cnstDefs.SYNTH1OUTLIER_CORRECTLYLABELED,...%        = 47;
        cnstDefs.SYNTH1OUTLIER_CORRECTLYLABELED_FURTHER,...%  = 48;
        cnstDefs.SYNTH1OUTLIER_CORRECTLYLABELED_FURTHER2,...% = 49;
        cnstDefs.SYNTH6OUTLIER_NEAR,...%                      = 50;
        cnstDefs.SYNTHTESTSMO,...%                            = 51;
        cnstDefs.SYNTHLABELNOISE,...%                         = 52;
        cnstDefs.SYNTH2OUTLIER_INCORRECTLYLABELED,...%        = 53;
        cnstDefs.SYNTH3OUTLIER_CORRECTLYLABELED,...%          = 54;
        cnstDefs.SYNTH4OUTLIER_CORRECTLYLABELED}
        clear 'XClass';
        data = getSynthDataset(datasetId);
    case {cnstDefs.DIST_DATASET_GAUSSIAN_ONEDIM} % THESE ARE DISTRIBUTIONAL DATASETS
        switch(datasetId)
            case cnstDefs.DIST_DATASET_GAUSSIAN_ONEDIM
               datafile ='gaussiandistTest.mat';
               load(datafile);
               data.Y = Z;
               data.X = X; % each element is an struct which itself is a list of vectors
               data.dist.d = X(1).d;
               data.d = Inf;
               data.n = numel(data.Y);
        end
        data.datasetName         = datasetName;
    case cnstDefs.BREAST  % Breast Cancer Dataset
        datasetName = 'Breast Cancer';
        data_dir = 'Datasets/BreastCancer/breast-cancer';
        data_dir = strcat(datasetmain,data_dir);
        % Load the raw data
        data.raw = load([data_dir, '/', 'breast-cancer.data']);
        % Get the number of rows (examples) in the raw data 
        data.n = size(data.raw, 1);
        data.n_al = min(data.n,40);
        % Get the number of cols (features) in the raw data.  Note that the last
        % column is the outcome and is not a feature
        data.d = size(data.raw, 2) - 1;
        % Create the covariate matrix of features
        data.X = data.raw(:, 1:data.d)';
        % Create the vector of outcomes where 4 ==> 1 and 2 ==> 0
        data.Y = data.raw(:, data.d + 1) == 4;
        data.Y = data.Y'*2-1;
        data.datasetName         = datasetName;
    case cnstDefs.BUPALIVER % Bupa Liver Dataset
        datasetName = 'Bupa Liver';
        data_dir = 'Datasets/BupaLiverDataset';
        data_dir = strcat(datasetmain,data_dir);
        % Load the raw data
        data.raw = load([data_dir, '/', 'bupa.data']);
        % Get the number of rows (examples) in the raw data 
        data.n = size(data.raw, 1);
        data.n_al = min(data.n,100);
        % Get the number of cols (features) in the raw data.  Note that the last
        % column is the outcome and is not a feature
        data.d = size(data.raw, 2) - 1;
        % Create the covariate matrix of features
        data.X = data.raw(:, 1:data.d)';
        % Create the vector of outcomes where 4 ==> 1 and 2 ==> 0
        data.Y = data.raw(:, data.d + 1) == 2;
        data.Y = data.Y'*2-1;     
        data.datasetName         = datasetName;
    case cnstDefs.ISOLET_1234 % isolet 1+2+3+4
        datasetName = 'isloet 1+2+3+4';
        data_dir = 'Datasets/isloet/isolet1+2+3+4.data';
        data_dir = strcat(datasetmain,data_dir);
        % Load the raw data
        load_original = true;
        if load_original
            data_raw = load([data_dir, '/', 'isolet1+2+3+4.data']);
            % Get the number of rows (examples) in the raw data 
            % I must know the structure of this dataset
            data.n = size(data_raw, 1);
            data.n_al = min(data.n,100);
            % Get the number of cols (features) in the raw data.  Note that the last
            % column is the outcome and is not a feature
            data.d = size(data_raw, 2) - 1;
            % Create the covariate matrix of features
            data.X = data_raw(:, 1:data.d);
            % Create the vector of outcomes where 4 ==> 1 and 2 ==> 0
            data.Y = data_raw(:, data.d + 1) == 2;
            data.Y = data.Y*2-1;     
            data.X  = data.X';
            data.Y  = data.Y';
            save([data_dir, '/', 'isolet1234_data.mat'],'data');
        else
            load([data_dir, '/', 'isolet1234_data.mat'],'data');
        end
        subdataset_method = 2;
        DS_settings{subdataset_method}.percent = 10;
        data.datasetName         = datasetName;
    case cnstDefs.ISOLET_5 % isolet 5
        datasetName = 'isolet 5';
        data_dir = 'Datasets/isloet/isolet5.data';
        data_dir = strcat(datasetmain,data_dir);
        % Load the raw data
        data_raw = load([data_dir, '/', 'isolet5.data']);
        % Get the number of rows (examples) in the raw data 
        % I must know the structure of this dataset
        data.n = size(data_raw, 1);
        data.n_al = min(data.n,30);
        % Get the number of cols (features) in the raw data.  Note that the last
        % column is the outcome and is not a feature
        data.d = size(data_raw, 2) - 1;
        % Create the covariate matrix of features
        data.X = data_raw(:, 1:data.d)';
        % Create the vector of outcomes where 4 ==> 1 and 2 ==> 0
        data.Y = data_raw(:, data.d + 1) == 2;
        data.Y = data.Y'*2-1;     
        subdataset_method = 2;
        DS_settings{subdataset_method}.percent = 20;
        data.datasetName         = datasetName;
    case {cnstDefs.LETTER_EvsD, cnstDefs.LETTER_PvsD, cnstDefs.LETTER_EvsF,...
          cnstDefs.LETTER_IvsJ, cnstDefs.LETTER_MvsN, cnstDefs.LETTER_UvsV}
        [data, DS_settings, gamma_ratio]  = getLetterDataset(datasetId, DS_settings);
    case cnstDefs.MNIST% MNIST data set
        datasetName = 'MNIST';
        data_dir='Datasets/MNIST/';
        data_dir=strcat(datasetmain,data_dir);
        addpath(data_dir);
        data_trImg = strcat(data_dir,'train-images.idx3-ubyte');
        data_teLab = strcat(data_dir,'train-labels.idx1-ubyte');
        data.X  =loadMNISTImages(data_trImg);
        data.class =loadMNISTLabels(data_teLab);
        classA    = 0;
        classB    = 1;
        data.n = size(data.X, 2);
        data.n_al = min(data.n,100);
        % Get the number of cols (features) in the raw data.  Note that the last
        % column is the outcome and is not a feature
        data.d    = size(data.X, 1); 
        [data]    = select_class_dataset(data, classA, classB);
        data.datasetName         = datasetName;
    case cnstDefs.VEHICLE% vehicle ,UCI
        classOne = 1;
        classMinusOne = 3;
        [data]  = getVehicleDataset( datasetmain, classOne, classMinusOne, DS_settings);
    case cnstDefs.IONOSHPERE % ionosphere ,UCI
         datasetName = 'ionospere';
        data_dir = 'Datasets/ionosphere/';
        data_dir=strcat(datasetmain,data_dir);
        data_file=strcat(data_dir,'ionosphere.txt');
        T1 = readtable(data_file,'Delimiter','comma','ReadVariableNames',false);
        data_file=strcat(data_dir,'xac.dat');
        data.d = size(T1,2)-1;
        data.n = size(T1,1);
        data.n_al = min(data.n,100);
        data.X  = table2array(T1(:,1:data.d-1));
        st      = cell2mat(table2array(T1(:,data.d+1)));
        data.Y  = st=='g';
        data.Y  = data.Y*2-1;
        data.X  = data.X';
        data.Y  = data.Y';
        data.datasetName         = datasetName;
    case {cnstDefs.SSLBOOK_DIGIT1, cnstDefs.SSLBOOK_USPS, cnstDefs.SSLBOOK_COIL2,cnstDefs.SSLBOOK_BCI,...
          cnstDefs.SSLBOOK_G241C , cnstDefs.SSLBOOK_G241N, cnstDefs.SSLBOOK_COIL, cnstDefs.SSLBOOK_TEXT }
        [data, DS_settings, gamma_ratio]  = getSSLDataset(datasetId, DS_settings);
        
    case cnstDefs.ECOLI
        datasetName = 'ECOLI_UCI';
        data_dir = 'Datasets/ecoli/';
        data_dir=strcat(datasetmain,data_dir);
        data_file=strcat(data_dir,'ecoli.dat');
        [data] = ecoliimportfile(data_file);
        data.class          = data.Y';
        data.X              = data.X';
        data.n              = size(data.X,2);
        data.n_al = min(data.n,40);
        classOne      = 1;    %classes are 1,2,3,4,5,6,7,8 
        classMinusOne = 2;
        [data] = select_class_dataset(data, classOne, classMinusOne);
        data.datasetName         = datasetName;
    case cnstDefs.GLASS
        datasetName = 'GLASS_UCI';
        data_dir = 'Datasets/glass/';
        data_dir=strcat(datasetmain,data_dir);
        data_file=strcat(data_dir,'glass.data');
        [data] = glassimportfile(data_file);
        data.class = data.Y;
        data.X     = data.X';
        data.n              = size(data.X,2);
        data.n_al = min(data.n,50);

        classOne      = 1;    %classes are 1,2,3,4,5,6,7 
        classMinusOne = 2;
        [data]        = select_class_dataset(data, classOne, classMinusOne);
        data.datasetName         = datasetName;
     case cnstDefs.HEART_STATLOG
        datasetName = 'HEART_STATLOG';
        data_dir = 'Datasets/Heart_statlog/';
        data_dir=strcat(datasetmain,data_dir);
        data_file=strcat(data_dir,'heart.dat');
        [data] = heart_importfile(data_file);
        data.X   = data.X';
        data.n              = size(data.X,2);
        data.n_al = min(data.n,150);
        data.datasetName         = datasetName;
     case cnstDefs.IMAGESEGMENT_STATLOG
        datasetName = 'IMAGESEGMENT_STATLOG';
        data_dir = 'Datasets/Image Segmentation_statlog/';
        data_dir=strcat(datasetmain,data_dir);
        data_file=strcat(data_dir,'segment.dat');
        [data] = Image_segmentation_statlog(data_file);
        data.class = data.Y';
        data.X   = data.X';
        classOne      = 1;    %classes are 1,2,3,4,5,6,7 
        classMinusOne = 2;
        [data]        = select_class_dataset(data, classOne, classMinusOne);
        data.n              = size(data.X,2);
        data.n_al = min(data.n,30);
        data.datasetName         = datasetName;
     case cnstDefs.PIMA_DIABETES
        datasetName = 'PIMA_DIABETES';
        data_dir = 'Datasets/\OUTLIERDETECTIONDATASETS\ODDS\PIMA\';
        data_dir=strcat(datasetmain,data_dir);
        data_file=strcat(data_dir,'pima-indians-diabetes.data');
        [data] = PIMA_importfile(data_file);
        data.Y = 2*data.Y-1;
        data.class = data.Y';
        data.X   = data.X';
        data.n              = size(data.X,2);
        data.n_al = min(data.n,60);
       data.datasetName         = datasetName;
     case cnstDefs.SATELLITE
        datasetName = 'SATELLITE';
        data_dir   = 'Datasets/\OUTLIERDETECTIONDATASETS\ODDS\SATELLITE\';
        data_dir   = strcat(datasetmain,data_dir);
        data_file  = strcat(data_dir,'sat.trn');
        [data]     = SATELLITE_importfile(data_file);
        data_file  = strcat(data_dir,'sat.tst');
 %       data_test =  SATELLITE_importfile(data_file);
        data.class = data.Y';
        data.X     = data.X';
        data.n              = size(data.X,2);
        data.n_al  = min(data.n,100);
        
        classOne      = 1;    %classes are 1,2,3,4,5,6,7 
        classMinusOne = 2;
        [data]        = select_class_dataset(data, classOne, classMinusOne);
        subdataset_method = 2;
        DS_settings{subdataset_method}.percent = 30;
        data.datasetName         = datasetName;
%         data.data_test = data_test;
    case cnstDefs.BREAST_WPDC
        datasetName = 'BREAST_WPDC';
        data_dir = 'Datasets/\OUTLIERDETECTIONDATASETS\ODDS\WBREAST\';
        data_dir=strcat(datasetmain,data_dir);
        data_file=strcat(data_dir,'wpbc.data');
        [data] = wpdc_importfile(data_file);
        data.X   = data.X';
        classOne      = 'N';    
        classMinusOne = 'R';
        [data]        = select_class_dataset(data, classOne, classMinusOne);
        data.n              = size(data.X,2);
        data.n_al = min(data.n,60);
        data.datasetName         = datasetName;
    case cnstDefs.CODRNA
        data = load_codrna(datasetmain,200);
        data.n_al = min(data.n , 50);
    case cnstDefs.COLON 
        data = load_colon(datasetmain,200);
        data.n_al = min(data.n , 100);
    case cnstDefs.COVTYPE 
        data = load_covtype(datasetmain,200);
        data.n_al = min(data.n , 50);
    case cnstDefs.FOURCLASS 
        data = load_fourclass(datasetmain,200);
        data.n_al = min(data.n , 50);
    case cnstDefs.GERMANNUMER 
        data = load_germannumer(datasetmain,200);
        data.n_al = min(data.n , 50);    
    case cnstDefs.GISETTE 
        data = load_giesette(datasetmain,200);
        data.n_al = min(data.n , 50);    
    case cnstDefs.IJCNN1 
        data = load_ijcnn(datasetmain,200);
        data.n_al = min(data.n , 50);   
    case cnstDefs.LIVERDISORDER
        data = load_liverdisorder(datasetmain,200);
        data.n_al = min(data.n , 50);
    case cnstDefs.MADELON
        data = load_madelon(datasetmain,200);
        data.n_al = min(data.n , 50);    
    case cnstDefs.MUSHROOM
        data = load_mushrooms(datasetmain,200);
        data.n_al = min(data.n , 50);
    case cnstDefs.NEWS20BINARY % with a huge number of features
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\news20.binary\news20.binary',...
                                          'news20.binary', 200);
        data.n_al = min(data.n , 50);        
    case cnstDefs.PHISHING 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\phishing\phishing',...
                                          'phishing', 200, 0, -1);
        data.n_al = min(data.n , 50);    
    case cnstDefs.REALSIM 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\real-sim\real-sim',...
                                          'real-sim', 200, 0, -1);
        data.n_al = min(data.n , 50);     
    case cnstDefs.SKINNOSKIN 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\skin_nonskin\skin_nonskin',...
                                          'skin_nonskin', 200, 2, -1);
        data.n_al = min(data.n , 50);  
    case cnstDefs.sonar 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\sonar_scale\sonar_scale',...
                                          'sonar', 200, 2, -1);
        data.n_al = min(data.n , 50);
    case cnstDefs.svmguide1 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\svmguide1\svmguide1',...
                                          'svmguide1', 200, 0, -1);
        data.n_al = min(data.n , 50);
    case cnstDefs.svmguide3 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\svmguide3\svmguide3',...
                                          'svmguide3', 200);
        data.n_al = min(data.n , 50);        
    case cnstDefs.w1a 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\w1a\w1a',...
                                          'w1a', 80);
        data.n_al = min(data.n , 30);       
    case cnstDefs.w2a 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\w2a\w2a',...
                                          'w2a', 200);
        data.n_al = min(data.n , 50);        
    case cnstDefs.w3a 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\w3a\w3a',...
                                          'w3a', 200);
        data.n_al = min(data.n , 50);       
    case cnstDefs.w4a 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\w4a\w4a',...
                                          'w4a', 200);
        data.n_al = min(data.n , 50);
        case cnstDefs.w5a 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\w5a\w5a',...
                                          'w5a', 200);
        data.n_al = min(data.n , 50);
    case cnstDefs.w6a 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\w6a\w6a',...
                                          'w6a', 200);
        data.n_al = min(data.n , 50);        
    case cnstDefs.w7a 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\w7a\w7a',...
                                          'w7a', 200);
        data.n_al = min(data.n , 50);
    case cnstDefs.w8a 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\w8a\w8a',...
                                          'w8a', 200);
        data.n_al = min(data.n , 50);        
    case cnstDefs.a1a 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\a1a\a1a',...
                                          'a1a', 200, 2, -1);
        data.n_al = min(data.n , 50);        
    case cnstDefs.a2a 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\a2a\a2a',...
                                          'a2a', 200, 2, -1);
        data.n_al = min(data.n , 50);        
    case cnstDefs.a3a 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\a3a\a3a',...
                                          'a3a', 200, 2, -1);
        data.n_al = min(data.n , 50);        
    case cnstDefs.a4a 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\a4a\a4a',...
                                          'a4a', 200, 2, -1);
        data.n_al = min(data.n , 50);        
    case cnstDefs.a5a 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\a5a\a5a',...
                                          'a5a', 200, 2, -1);
        data.n_al = min(data.n , 50);        
    case cnstDefs.a6a 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\a6a\a6a',...
                                          'a6a', 200, 2, -1);
        data.n_al = min(data.n , 50);        
    case cnstDefs.a7a 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\a7a\a7a',...
                                          'a7a', 200, 2, -1);
        data.n_al = min(data.n , 50);        
    case cnstDefs.a8a 
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\a8a\a8a',...
                                          'a8a', 200, 2, -1);
        data.n_al = min(data.n , 50);        
    case cnstDefs.a9a
        data = load_libsvmdataset_general(datasetmain,'\LIBSVM-Binary\a9a\a9a',...
                                          'a9a', 200, 2, -1);
        data.n_al = min(data.n , 50);        
end
data.F                   = 1:data.n;
data.F_id                = 1:data.n;
data.LB_mapping_func     = LB_mapfunc_list{LB_mapping_method};
data.LB_settings         = LB_settings_list{LB_mapping_method}; 
[data          ]         = data.LB_mapping_func(data, data.LB_settings);
[data.learningparams_init ] = compute_learningparams(data, datasetId,gamma_ratio, gamma_o_ratio, lambda_o_ratio,kernel_func);
data.DS_subsampling_func = DS_subsampling_list{subdataset_method};
data.DS_settings         = DS_settings{subdataset_method};
data.TR_sampling_func    = TR_subsampling_list{TR_subdataset_method};
data.TR_settings         = TR_settings{TR_subdataset_method};
data.UN_sampling_func    = UN_labeled_subset_query_list{UN_subsampling_method};
data.UN_settings         = UN_settings{UN_subsampling_method};

data.isDistData          = false;
if profile.LOAD_SAVE_SVM_PARAMS % if it not exist compute it and store it. 
    fname = get_SVMparamsfilename(data_dir, datasetName);
    recompute = true;
    if exist(fname,'file') && ~recompute
       load(fname,'learning_params');
    else
       Classification_exp_param = set_classification_experiment('SVM' ,@SVMtrain ,@SVMtester, @SVM_learning_list_maker);                             
       temp_profile = profile;
       temp_profile.kfold = 2;
       temp_profile.showdata    = false;
       temp_profile.CV_search_randomselect = false;
       learning_params          = cross_validation(Classification_exp_param, data, temp_profile, learning_params_init);  
       save(fname,'learning_params');
    end
    data.learningparams_init = learning_params;
end
data.sigma = 1;
data.C   = unique(data.Y);
data.n_C = numel(data.C);
if ~isfield(data, 'binary')
    data.binary = true;
end
data.datasetId = datasetId;
if ~isfield(data, 'divide_func')
    data.divide_func = @divide_data_general;
end
if ~isfield(data, 'split_data_func')
    data.split_data_func = @split_data_general;
end
comp_mean = false;
if comp_mean
    data =  comp_mean_vec_dist(data);
    data.has_mean_vec = true;
end
data.has_mean_vec = comp_mean;
if ~isfield(data, 'data_comp_kernel')
    data.data_comp_kernel = @comp_simple_kernel_general;
end
end
function [data] = load_letter_dataset(data_file, LetA, LetB)
 % Load the raw data
    raw_data = readtable(data_file,'Delimiter','comma','ReadVariableNames',false,'FileEncoding','windows-1254');
    % Get the number of rows (examples) in the raw data 
    % I must know the structure of this dataset
    data.n = size(raw_data, 1);
    % Get the number of cols (features) in the raw data.  Note that the last
    % column is the outcome and is not a feature
    data.d    = size(raw_data, 2)-1 ; 
    data.lettername  = cell2mat(table2array(raw_data(:,1))); 
    data.X = table2array(raw_data(:,2:end)); 


    data.X = data.X(data.lettername==LetA | data.lettername==LetB,:);
    data.lettername = data.lettername(data.lettername==LetA | data.lettername==LetB);
    % Create the covariate matrix of features
    %data.X = data.raw(:, 2:data.d);
    % Create the vector of outcomes where 4 ==> 1 and 2 ==> 0
    data.Y = data.lettername == LetA;
    data.Y = data.Y*2-1;     
    data.n = size(data.X,1);
    data.ClassOne = LetA;
    data.ClassminusOne = LetB;
end
function [data] = select_class_dataset(data, classOne, classMinusOne)
    data.class= data.class';
    data.X = data.X(:,data.class==classOne | data.class==classMinusOne);
    data.class = data.class(data.class==classOne | data.class==classMinusOne);
    % Create the covariate matrix of features
    %data.X = data.raw(:, 2:data.d);
    % Create the vector of outcomes where 4 ==> 1 and 2 ==> 0
    data.Y = data.class == classOne;
    data.Y = data.Y*2-1;     
    data.n = size(data.X,2);
    data.ClassOne = classOne;
    data.ClassminusOne = classMinusOne;
end
function [learning_params_init] = compute_learningparams(data, datasetId, gamma_ratio, gamma_o_ratio, lambda_o_ratio,kernel_func)
global cnstDefs
    switch(datasetId)
        case cnstDefs.BREAST  % Breast Cancer Dataset
        case cnstDefs.BUPALIVER % Bupa Liver Dataset
        case cnstDefs.ISOLET_1234 % isolet 1+2+3+4
        case cnstDefs.ISOLET_5 % isolet 5
        case {cnstDefs.LETTER_EvsD, cnstDefs.LETTER_PvsD, cnstDefs.LETTER_EvsF,...
              cnstDefs.LETTER_IvsJ, cnstDefs.LETTER_MvsN, cnstDefs.LETTER_UvsV}
            switch(datasetId)
                case cnstDefs.LETTER_PvsD
                case cnstDefs.LETTER_EvsD
                case cnstDefs.LETTER_EvsF
                case cnstDefs.LETTER_IvsJ
                case cnstDefs.LETTER_MvsN
                case cnstDefs.LETTER_UvsV
            end
        case cnstDefs.MNIST% MNIST data set
        case cnstDefs.VEHICLE% vehicle ,UCI
        case cnstDefs.IONOSHPERE % ionosphere ,UCI
        case {cnstDefs.SSLBOOK_DIGIT1, cnstDefs.SSLBOOK_USPS, cnstDefs.SSLBOOK_COIL2,cnstDefs.SSLBOOK_BCI,...
              cnstDefs.SSLBOOK_G241C , cnstDefs.SSLBOOK_G241N, cnstDefs.SSLBOOK_COIL, cnstDefs.SSLBOOK_TEXT }
            switch (datasetId)
                case cnstDefs.SSLBOOK_DIGIT1
                case cnstDefs.SSLBOOK_USPS
                case cnstDefs.SSLBOOK_COIL2
                case cnstDefs.SSLBOOK_BCI
                case cnstDefs.SSLBOOK_G241C
                case cnstDefs.SSLBOOK_G241N
                case cnstDefs.SSLBOOK_COIL
                case cnstDefs.SSLBOOK_TEXT
            end
        case cnstDefs.ECOLI
        case cnstDefs.GLASS
        case cnstDefs.HEART_STATLOG
        case cnstDefs.IMAGESEGMENT_STATLOG
        case cnstDefs.PIMA_DIABETES
        case cnstDefs.SATELLITE
        case cnstDefs.BREAST_WPDC
    end
    lambda   = 1;
    lambda_o = lambda/lambda_o_ratio;
    n        = numel(data.Y);
    mediandist = median(reshape(pdist2(data.X',data.X'),n^2,1));
    gamma    = gamma_ratio/(mediandist^2);
    gamma_is = gamma;
    gamma_o  = gamma/gamma_o_ratio;
    sigma_likelihood = 1;
    [learning_params_init ]  = learning_settings(data.n, 'gamma'      , gamma, 'gamma_is', gamma_is, 'gamma_o', gamma_o,...
                                                         'kernel_func', kernel_func,...
                                                         'lambda'     , lambda, 'lambda_o', lambda_o, 'sigma_likelihood',sigma_likelihood,...
                                                         'data_noisy' , data.noisy, 'data_labelnoise', data.labelnoise); 
end
function  [data]  = getVehicleDataset( datasetmain, classOne, classMinusOne, DS_settings)

        
        data_dir = 'Datasets/vehicle/';
        data_dir=strcat(datasetmain,data_dir);
        data_file=strcat(data_dir,'xaa.dat');
        T1 = readtable(data_file,'Delimiter',' ','ReadVariableNames',false);
        data_file=strcat(data_dir,'xac.dat');
        data.d = 19;
        T1 = T1(:,1:data.d);
        T2 = readtable(data_file,'Delimiter',' ','ReadVariableNames',false);
        T2 = T2(:,1:data.d);
        data_file=strcat(data_dir,'xad.dat');
        T3 = readtable(data_file,'Delimiter',' ','ReadVariableNames',false);
        T3 = T3(:,1:data.d);
        data_file=strcat(data_dir,'xae.dat');
        T4 = readtable(data_file,'Delimiter',' ','ReadVariableNames',false);
        T4 = T4(:,1:data.d);
        data_file=strcat(data_dir,'xaf.dat');
        T5 = readtable(data_file,'Delimiter',' ','ReadVariableNames',false);
        T5 = T5(:,1:data.d);
        data_file=strcat(data_dir,'xag.dat');
        T6 = readtable(data_file,'Delimiter',' ','ReadVariableNames',false);
        T6 = T6(:,1:data.d);
        data_file=strcat(data_dir,'xah.dat');
        T7 = readtable(data_file,'Delimiter',' ','ReadVariableNames',false);
        T7 = T7(:,1:data.d);
        data_file=strcat(data_dir,'xai.dat');
        T8 = readtable(data_file,'Delimiter',' ','ReadVariableNames',false);
        T8 = T8(:,1:data.d);
        data.raw  = [T1;T2;T3;T4;T5;T6;T7;T8];
        data.d    = size(data.raw, 2) - 1;
        data.n    = size(data.raw, 1);
        data.n_al = min(data.n,100);
        data.lettername  = table2array(data.raw(:,data.d+1)); 
        data.X = table2array(data.raw(:,1:data.d));
        % convert lettername to class number ,or logical 
        carsname={'opel','saab','bus','van'};
        idx_ltname = reshape(1:numel(data.lettername), size(data.lettername));
        idx_carsname = reshape(1:numel(carsname), size(carsname));
        data.logicalClass = bsxfun(@(ii,jj)strcmp(data.lettername(ii),carsname(jj)), idx_ltname, idx_carsname);
        data.classnum     = zeros(data.n,1);
        data.class(data.logicalClass(:,1))=1;
        data.class(data.logicalClass(:,2))=2;
        data.class(data.logicalClass(:,3))=3;
        data.class(data.logicalClass(:,4))=4;
        data.class = data.class';
        datasetName = 'vehicle';
        datasetName = [datasetName, carsname{classOne}, carsname{classMinusOne}];
        data.X        = data.X';
        data.logicalClass = data.logicalClass;
        data.datasetName = datasetName;
        [data] = select_class_dataset(data, classOne, classMinusOne);
end
function  [data, DS_settings, gamma_ratio]  = getLetterDataset(datasetId, DS_settings)

        switch(datasetId)
            case cnstDefs.LETTER_PvsD
                datasetName = 'P vs D';
                LetA = 'P';
                classB = 'D';
            case cnstDefs.LETTER_EvsD
                datasetName = 'E vs D';
                LetA = 'E';
                classB = 'D';
            case cnstDefs.LETTER_EvsF
                datasetName = 'E vs D';
                LetA = 'E';
                classB = 'F';
            case cnstDefs.LETTER_IvsJ
                datasetName = 'I vs J';
                LetA = 'I';
                classB = 'J';
            case cnstDefs.LETTER_MvsN
                datasetName = 'M vs N';
                LetA = 'M';
                classB = 'N';
            case cnstDefs.LETTER_UvsV
                datasetName = 'U vs V';
                LetA = 'U';
                classB = 'V';
        end        
        data_dir = 'Datasets/Letter/';
        data_dir = strcat(datasetmain,data_dir);
        load_original = true;
        if load_original 
            data_file = strcat(data_dir,'letter-recognition.txt');
            [data]    = load_letter_dataset(data_file, LetA, classB);
            data.X    = data.X';
            data.Y    = data.Y';
            save([data_dir, '/', datasetName,'.mat'],'data');
        else
            load([data_dir, '/', datasetName,'.mat'],'data');
        end
        data.n    = size(data.X,2);
        data.n_al = min(data.n,100);
        gamma_ratio       = 2;
        subdataset_method = 2;
        DS_settings{subdataset_method}.percent = 40;
end
function  [data, DS_settings, gamma_ratio]  = getSSLDataset(datasetId, DS_settings)

        switch (datasetId)
            case cnstDefs.SSLBOOK_DIGIT1
                datasetName   = 'Digit1';
                data_dir      = 'Datasets/SSLBookBenchMarkDatasets/Digit1-set1/';
                data_fname    = 'SSL,set=1,data.mat';
            case cnstDefs.SSLBOOK_USPS
                datasetName   = 'USPS';
                data_dir      = 'Datasets/SSLBookBenchMarkDatasets/USPS-set2/';
                data_fname    = 'SSL,set=2,data.mat';
            case cnstDefs.SSLBOOK_COIL2
                datasetName   = 'COIL2';
                data_dir      = 'Datasets/SSLBookBenchMarkDatasets/COIL2-set3/';
                data_fname    = 'SSL,set=3,data.mat';
            case cnstDefs.SSLBOOK_BCI
                datasetName   = 'BCI';
                data_dir      = 'Datasets/SSLBookBenchMarkDatasets/BCI-set4/';
                data_fname    = 'SSL,set=4,data.mat';
            case cnstDefs.SSLBOOK_G241C
                datasetName   = 'g241c';
                data_dir      = 'SSLBookBenchMarkDatasets/g241c-set5/';
                data_fname    = 'SSL,set=5,data.mat';
            case cnstDefs.SSLBOOK_G241N
                datasetName   = 'g241n';
                data_dir      = 'Datasets/SSLBookBenchMarkDatasets/g241n-set7/';
                data_fname    = 'SSL,set=7,data.mat';
            case cnstDefs.SSLBOOK_COIL
                datasetName   = 'COIL';
                data_dir      = 'Datasets/SSLBookBenchMarkDatasets/COIL-set6/';
                data_fname    = 'SSL,set=6,data.mat';
                lam           = 0.1;
            case cnstDefs.SSLBOOK_TEXT
                datasetName   = 'TEXT';
                data_dir      = 'Datasets/SSLBookBenchMarkDatasets/Text-set9/';
                data_fname    = 'SSL,set=9,data.mat';
        end
        data_dir    = strcat(datasetmain,data_dir);
        data_file   = strcat(data_dir,data_fname);
        load(data_file);
        data.X      = X';
        data.Y      = y';
        data.Y      = data.Y*2-1;
        data.n      = size(data.X,2);
        data.n_al = min(data.n,100);
        data.datasetName = datasetName;
        gamma_ratio    = 8;
        subdataset_method = 2;
        DS_settings{subdataset_method}.percent = 30;
end
function  data = getSynthDataset(datasetId)
        switch(datasetId)
            case cnstDefs.SYNTH4OUTLIERORDERED 
               datafile ='Copy_of_outlier4orderedlittle.mat';
               load(datafile, 'XClass');
               XClass = XClass';
               datasetName = 'SYNTH4OUTLIERORDERED';    
            case cnstDefs.SYNTHORDEREDOUTLIERLARGER
               datafile ='order4outlierLambdaExp';
               load(datafile);
               XClass = XClass';
               datasetName = 'ORDEREDOUTLIERLARGER';   
            case cnstDefs.SYNTHORDEREDLITTLE_WITHOUTOUTLIER 
               datafile ='orderedlittle';
               load(datafile);
               XClass = XClass';
               datasetName = 'ORDEREDLITTLE_WITHOUTOUTLIER'; 
            case cnstDefs.SYNTHORDERED_DENSE_TWOOUTLIERS 
               datafile ='orderedlittleOutlier2t1';
               load(datafile);
               XClass = XClass';
               datasetName = 'SYNTHORDERED_DENSE_TWOOUTLIERS';  
            case cnstDefs.SYNTHORDERED_DENSE_ONEOUTLIERS 
               datafile ='orderedlittleOutlier2t2';
               load(datafile);
               XClass = XClass';
               datasetName = 'SYNTHORDERED_DENSE_ONEOUTLIERS'; 
            case cnstDefs.SYNTHONEOUTLIER_ANOTHER 
               datafile ='orderedlittleOutlier2t2copy';
               load(datafile);
               XClass = XClass';
               datasetName = 'SYNTHONEOUTLIER_ANOTHER';
            case cnstDefs.SYNTH2OUTLIER_FARAWAY 
               datafile ='orderedOutlier2t4';
               load(datafile);
               XClass = XClass';
               datasetName = 'SYNTH2OUTLIER_FARAWAY';   
            case cnstDefs.SYNTH4OUTLIER_INTHESAMEDIRECTION
               datafile ='orderedOutlier2t5';
               load(datafile);
               XClass = XClass';
               datasetName = 'SYNTH4OUTLIER_INTHESAMEDIRECTION';  
            case cnstDefs.SYNTH4OUTLIER_TWOINBOUNDRY
               datafile ='outlier4orderedlittle';
               load(datafile);
               XClass = XClass';
               datasetName = 'SYNTH4OUTLIER_TWOINBOUNDRY'; 
            case cnstDefs.SYNTH4OUTLIER_TWOINBOUNDRY_MORELARGER
               datafile ='outlier4orderedlittleplusmanydata';
               load(datafile);
               XClass = XClass';
               datasetName = 'SYNTH4OUTLIER_TWOINBOUNDRY_MORELARGER';   
            case cnstDefs.SYNTH6OUTLIER_TWOINBOUNDRY_MORELARGER
               datafile ='outlier6orderedlittleplusmanydata';
               load(datafile);
               XClass = XClass';
               datasetName = 'SYNTH6OUTLIER_TWOINBOUNDRY_MORELARGER'; 
            case cnstDefs.SYNTH1OUTLIER_INBOUNDRYSMALL
               datafile ='Outlier_orderedlittle';
               load(datafile);
               XClass = XClass';
               datasetName = 'SYNTH1OUTLIER_INBOUNDRYSMALL';   
            case cnstDefs.SYNTH1OUTLIER_INBOUNDRYSMALL_FURTHER
               datafile ='Outlierorderedlittle';
               load(datafile);
               XClass = XClass';
               datasetName = 'SYNTH1OUTLIER_INBOUNDRYSMALL_FURTHER';  
            case cnstDefs.SYNTH1OUTLIER_CORRECTLYLABELED
               datafile ='testSCCOutsidesame';
               load(datafile);
               XClass = XClass';
               datasetName = 'SYNTH1OUTLIER_CORRECTLYLABELED';    
            case cnstDefs.SYNTH1OUTLIER_CORRECTLYLABELED_FURTHER
               datafile ='testSimplef';
               load(datafile);
               XClass = XClass';
               datasetName = 'SYNTH1OUTLIER_CORRECTLYLABELED_FURTHER';  
            case cnstDefs.SYNTH1OUTLIER_CORRECTLYLABELED_FURTHER2
               datafile ='testSimplef2';
               load(datafile);
               XClass = XClass';
               datasetName = 'SYNTH1OUTLIER_CORRECTLYLABELED_FURTHER2'; 
            case cnstDefs.SYNTH6OUTLIER_NEAR
               datafile ='testSimplef3';
               load(datafile);
               XClass = XClass';
               datasetName = 'SYNTH1OUTLIER_NEAR';   
            case cnstDefs.SYNTHTESTSMO
               datafile ='testsmo';
               load(datafile);
               XClass = [trainPoints';trainLabels'];
               datasetName = 'SYNTHTESTSMO';     
            case cnstDefs.SYNTHLABELNOISE
               datafile ='testtwofuncCCSC';
               load(datafile);
               XClass = XClass';
               datasetName = 'SYNTHLABELNOISE';  
            case cnstDefs.SYNTH2OUTLIER_INCORRECTLYLABELED
               datafile ='testtwofuncOutSideInv';
               load(datafile);
               XClass = XClass';
               datasetName = 'SYNTH2OUTLIER_INCORRECTLYLABELED'; 
            case cnstDefs.SYNTH3OUTLIER_CORRECTLYLABELED
               datafile ='testtwofuncOutsideSame2point';
               load(datafile);
               XClass = XClass';
               datasetName = 'SYNTH3OUTLIER_CORRECTLYLABELED';    
            case cnstDefs.SYNTH4OUTLIER_CORRECTLYLABELED
               datafile ='testtwofuncOutsideSame4point';
               load(datafile);
               XClass = XClass';
               datasetName = 'SYNTH4OUTLIER_CORRECTLYLABELED';    
            case cnstDefs.SYNTHLINEARSVM 
               datafile ='linearUnlabeledsvm';
               load(datafile);
               XClass = XClass';
               datasetName = 'linearUnlabeledsvm';  
            case cnstDefs.SYNTHWITHOUTOUTLIERORDERED
               datafile ='Copy_of_orderedlittle.mat';
               load(datafile, 'XClass');
               XClass = XClass';
               datasetName = 'SYNTH_WITHOUTOUTLIER_ORDERED';   
            case cnstDefs.SYNTHMORECOMPLEXTESTSIMPLEFUNC
               datafile ='moreComplextestSimplef2';
               load(datafile);
               XClass = XClass';
               datasetName = 'MORECOMPLEXTESTSIMPLEFUNC';  
               
        end
        data.d = 2;
        data.n = size(XClass,2);
        data.n_al = min(n,50);
        data.X = XClass(1:2,:);
        data.Y = XClass(3  ,:);
        data.datasetName = datasetName;
end
%     case cnstDefs.ADULT
%         % not working 
%         datasetName = 'Adult';
%         data_dir = 'Adult/';
%         data_dir=strcat(datasetmain,data_dir);
%         data_file=strcat(data_dir,'adult_clean.dat');
%         T1 = load(data_file);
%         data.raw  = T1;
%         data.d    = size(data.raw, 2) - 1;
%         data.n    = size(data.raw, 1);
%         data.lettername  = table2array(data.raw(:,data.d+1)); 
%         data.X = table2array(data.raw(:,1:data.d));
%         % convert lettername to class number ,or logical 
%         
%         [data          ]    = LB_mapping_func(data, LB_settings); 
%         data.lambda = 1;
%         data.lambda_o = data.lambda/lambda_o_ratio;
%         data.gamma  = median(median(pdist2(data.X,data.X)))/sqrt(2);        
%         data.gamma_o= data.gamma/gamma_o_ratio;