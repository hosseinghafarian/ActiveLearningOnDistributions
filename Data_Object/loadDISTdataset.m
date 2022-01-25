function [data]=loadDISTdataset(datasetId, profile, recompute)
%% Load datasets based on Id
% the following list of functions, subsamples from dataset, i.e., it
% selects a subset of samples for all of the subsequent
% operations(including train, activelearning queries, test) 
% DS is abbreviation for dataset 
global cnstDefs
global crisisDInd

subdataset_method                   = profile.DSsubsampling_id;
DS_subsampling_list                 = {@DS_all_sampling,@DS_DISTRIBUTIONALrandom_sampling};
DS_all_sampling_settings.percent    = 100;
DS_random_sampling_settings.percent = profile.DSsubsampling_per;
DS_crossvalidation_settings.percent = profile.CVsubsampling_per;
DS_settings                         = {DS_all_sampling_settings, DS_random_sampling_settings, DS_crossvalidation_settings}; 
% the following list of functions, selects a subset of already selected
% samples from dataset for querying, i.e. it only selects a subset of instances from which we query.  
UN_subsampling_method               = profile.UNsubsampling_id;
UN_labeled_subset_query_list        = {@queryall_instances,@queryall_instances,};
UN_all_sampling_settings.percent    = 100;
UN_random_sampling_settings.percent = profile.UNsubsampling_per;
UN_settings                         = {UN_all_sampling_settings, UN_random_sampling_settings};
% the following list of functions, selects a subset of already selected
% instances for training set when the training set is large
TR_subdataset_method                = profile.TRsubsampling_id;
TR_subsampling_list                 = {@TS_all_sampling};
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
LB_mapfunc_list                     = {@LB_donothing
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

datasetmain= [cnstDefs.dataset_path,'/DISTDatasets/'];%strcat(cnstDefs.main_path,'/DISTDatasets/');
gamma_o_ratio  = 1;
gamma_ratio    = 1;
lambda_o_ratio = 10;

kernel_func    = @rbf_twolevel;

switch(datasetId)
    case {cnstDefs.DIST_SYNTH_MUGAMMA, cnstDefs.DIST_DATASET_GAUSSIAN_ONEDIM,...
          cnstDefs.DIST_SYNTH_MUANDETTOY1, cnstDefs.DIST_SYNTH_MUANDETTOY2,cnstDefs.DIST_SYNTH_MUANDETTOY3  }
        synthdata_subfolder = '\SYNTHESISDATA\';
        switch(datasetId)
            case cnstDefs.DIST_SYNTH_MUGAMMA
               mugamma1 = '\MUGAMMA_1\mugamma_9.mat';
               filename = strcat(strcat(datasetmain,synthdata_subfolder), mugamma1);
               n   = 80;
               nos_low = 5;
               nos_high= 200;
               d   = 3;
               recomp   = true;
               mu_range = [0, 10];
               sig_range= [ 0, 10];
               if ~recomp && exist(filename,'file')
                   load(filename,'data')
               else
                   data    = getDistData(mu_range,sig_range,n, d, nos_low, nos_high);
%                    data.gamma_is = median(median(pdist2(data.X',data.X')));
%                    data.dm = distance_matrix(data, data, data.gamma_is, true);
%                    [data.lambda, data.lambda_o, data.gamma, data.gamma_o ] = compute_DISTgamma(data, gamma_ratio, gamma_o_ratio, lambda_o_ratio);
%                    data.K  = comp_KernelEmb(data.dm, data.gamma);
                   save(filename, 'data');
               end
               data.noisy       = false(1,n);
               data.labelnoise  = false(1,n);
               data.datasetName = 'MUGAMMA1';
            case cnstDefs.DIST_DATASET_GAUSSIAN_ONEDIM   
                
            case cnstDefs.DIST_SYNTH_MUANDETTOY1 % this is a small dataset consist of 10 distributions
                mutoydata = '\MUANDET_TOY_DATA\muandet_toy_1.mat';
                filename  = strcat(strcat(datasetmain,synthdata_subfolder), mutoydata);
                recomp    = true;
                varsize   = false;  avgsize = 30; sigsize= 5;
                [data] = get_save_data(filename, @toy_data_set, 1,varsize, avgsize, sigsize, recomp, gamma_ratio, gamma_o_ratio, lambda_o_ratio);           
                
                data.datasetName = 'MU_TOY_1';
            case cnstDefs.DIST_SYNTH_MUANDETTOY2 % this is bigg
                mutoydata = '\MUANDET_TOY_DATA\muandet_toy_2.mat';
                filename  = strcat(strcat(datasetmain,synthdata_subfolder), mutoydata);
                recomp    = false;
                varsize   = true;  avgsize = 5; sigsize= 5;
                [data] = get_save_data(filename, @toy_data_set, 3, varsize, avgsize, sigsize ,recomp, gamma_ratio, gamma_o_ratio, lambda_o_ratio);       
                data.datasetName = 'SMMTOY_2';    
            case cnstDefs.DIST_SYNTH_MUANDETTOY3 % this is bigg
                mutoydata = '\MUANDET_TOY_DATA\muandet_toy_3.mat';
                filename  = strcat(strcat(datasetmain,synthdata_subfolder), mutoydata);
                recomp    = true;
                varsize   = true;  avgsize = 20; sigsize= 10;
                [data] = get_save_data(filename, @toy_data_set, 2, varsize, avgsize, sigsize ,recomp, gamma_ratio, gamma_o_ratio, lambda_o_ratio);       
                data.datasetName = 'SMMTOY_3'; 
        end
        data.n_al                = 50;
    case {cnstDefs.USPSDIST_6_9, cnstDefs.USPSDIST_3_8, cnstDefs.USPSDIST_1_8, cnstDefs.USPSDIST_3_4}
        switch(datasetId)
            case cnstDefs.USPSDIST_6_9
                digit1 = 9;       digit2 = 6;
                gensize = 2;     gensize_var = 10; 
                samplesize = 100;
                recomp = false;
                data   = load_USPDSDigits(datasetmain, digit1, digit2, samplesize, gensize,gensize_var, recomp) ;
            case cnstDefs.USPSDIST_3_8
                digit1 = 3;       digit2 = 8;
                gensize = 2;     gensize_var = 10; 
                samplesize = 100;
                recomp = false;
                data   = load_USPDSDigits(datasetmain, digit1, digit2, samplesize, gensize,gensize_var, recomp) ;
            case cnstDefs.USPSDIST_1_8
                digit1 = 1;       digit2 = 8;
                gensize = 2;     gensize_var = 10; 
                samplesize = 100;
                recomp = true;
                data   = load_USPDSDigits(datasetmain, digit1, digit2, samplesize, gensize,gensize_var, recomp) ;
            case cnstDefs.USPSDIST_3_4
                digit1 = 3;       digit2 = 4;
                gensize = 2;     gensize_var = 10; 
                samplesize = 100;
                recomp = true;
                data   = load_USPDSDigits(datasetmain, digit1, digit2, samplesize, gensize,gensize_var, recomp) ;
        end 
        data.n_al  = 50;
        data.F_id  = unique(data.F); 
    case cnstDefs.MUSK1
        muskfile = '\musk1_data_file.mat';
        muskfile = strcat(datasetmain, muskfile);
        recomp = true;
        if recomp 
            [data] = musk1_importfile('musk1.data', 1, 476);
            save(muskfile, 'data');
        else
            load(muskfile, 'data');
        end
        data.n_al                = 46;
     case cnstDefs.MUSK2
        muskfile = '\musk2_data_file.mat';
        muskfile = strcat(datasetmain, muskfile);
        recomp = true;
        if recomp 
            [data] = load_musk2('musk2norm_matlab', 'musk2.mat');
            save(muskfile, 'data');
        else
            load(muskfile, 'data');
        end
        data.n_al                = 60;
    case cnstDefs.NEWSGROUPRANGE
        datasetpath   = [datasetmain, '/20newsgroup/'];
        newsgroup_ind = datasetId - cnstDefs.NEWSGROUPSTART + 1; 
        fl            = cnstDefs.first_label(newsgroup_ind);
        sl            = cnstDefs.second_label(newsgroup_ind);
        dname         = ['20news_', sprintf('%02d',fl), sprintf('%02d',sl)];
        load([datasetpath, dname,'.mat'],'data');
        data.datasetName = dname;
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 50;
    case cnstDefs.NEWSGROUPONEVSRESTRANGE 
         datasetpath   = [datasetmain, '/20newsgroup/'];
        fl            = datasetId - cnstDefs.NEWSGROUPONEVSRESTSTART + 1; 
        dname         = ['20news_', sprintf('%02d',fl), 'vsrest'];
        load([datasetpath, dname,'.mat'],'data');
        data.datasetName = dname;
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 50;        
    case cnstDefs.CIFARRANGE
        datasetpath   = [datasetmain, '/CIFAR10/'];
        cifar_ind = datasetId - cnstDefs.CIFARSTART + 1; 
        fl            = cnstDefs.cifarfirst_label(cifar_ind);
        sl            = cnstDefs.cifarsecond_label(cifar_ind);
        dname         = ['cifar', sprintf('%1d',fl), sprintf('%1d',sl)];
        load([datasetpath, dname,'.mat'],'data');
        data.datasetName = dname;
        data.X           = double(data.X);
        
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 50;     
       case cnstDefs.CIFARMLCL
        datasetpath   = [datasetmain, '/CIFAR10/'];
        dname         = 'cifarMLCL';
        load([datasetpath, dname,'.mat'],'data');
        
        data.datasetName = dname;
        data.X           = double(data.X);
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 200;     
     case cnstDefs.STL10MLCL
        datasetpath   = [datasetmain, '/\ImageDataSets\stl10_matlab/'];
        dname         = 'STL-10';
        load([datasetpath, dname,'.mat'],'data');
        
        data.datasetName = dname;
        data.X           = double(data.X);
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 200;         
   case cnstDefs.REUTERSRANGE
        datasetpath      = [datasetmain, '/reuters/'];
        retuters_ind     = datasetId - cnstDefs.REUTERSSTART + 1; 
        fl               = cnstDefs.firreu_label(retuters_ind);
        sl               = cnstDefs.secreu_label(retuters_ind);
        dname            = ['reuters_', sprintf('%02d',fl), sprintf('%02d',sl)];
        load([datasetpath, dname,'.mat'],'data');
        data.datasetName = dname;
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 50;
   case cnstDefs.REUTERSMLCL
        datasetpath      = [datasetmain, '/reuters/'];
        dname            = 'reutersfinal.mat';
        load([datasetpath, dname],'data');
        %data.Y           = data.labelId;
        data.datasetName = dname;
        data.binary      = false;
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 50;    
    case cnstDefs.REUTERSONEVSRESTRANGE 
        datasetpath      = [datasetmain, '/reuters/'];
        retuters_ind     = datasetId - cnstDefs.REUTERSSTART + 1; 
        dname            = ['reuters_', sprintf('%02d',fl), 'vsrest'];
        load([datasetpath, dname,'.mat'],'data');
        data.datasetName = dname;
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 50;      
    case cnstDefs.YELPPOLARITY
        datasetpath      = [datasetmain, '/yelp_review_polarity/'];
        dname            = 'yelp_0102';
        load([datasetpath, dname,'.mat'],'data');
        data.datasetName = dname;
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 50;
    case cnstDefs.YELPSENT
        datasetpath      = [datasetmain, '/sentimentsentences/'];
        dname            = 'yelpsent_0001';
        load([datasetpath, dname,'.mat'],'data');
        data.datasetName = dname;
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 200;
    case cnstDefs.IMDBPOLARITY
        datasetpath      = [datasetmain, '/sentimentsentences/'];
        dname            = 'imdb_0001';
        load([datasetpath, dname,'.mat'],'data');
        data.datasetName = dname;
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 200; 
    case cnstDefs.AMAZONPOLARITY
        datasetpath      = [datasetmain, '/sentimentsentences/'];
        dname            = 'amazon_0001';
        load([datasetpath, dname,'.mat'],'data');
        data.datasetName = dname;
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 200;        
    case cnstDefs.COILUNRANGE
        datasetpath      = [datasetmain, '/coil/coil-20-unproc/'];
        coilun_ind         = datasetId - cnstDefs.COILUNSTART + 1; 
        fl               = cnstDefs.coilunfirst_label(coilun_ind);
        sl               = cnstDefs.coilunsecond_label(coilun_ind);
        dname            = ['unproc_coil_b_', sprintf('%d',fl),'_', sprintf('%d',sl)];
        load([datasetpath, dname,'.mat'],'data_b');
        data             = data_b;
        data.datasetName = dname;
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 50; 
        data.Y           = data.Y';
    case cnstDefs.COILUNMLCL
        datasetpath   = [datasetmain, '/coil/coil-20-unproc/'];
        dname         = 'unproc_coil_mlclass';
        load([datasetpath, dname,'.mat'],'data');
        data.binary      = false;
        data.datasetName = dname;
        data.X           = double(data.X);
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 50;     
    case cnstDefs.COIL20MLCL
        datasetpath      = [datasetmain, '/coil/coil-20-proc/'];
        dname            = 'proc_coil_mlclass'; % when we randomly select
        % images the result is that accuracy even with 2 distrbutional
        % examples labeled is about 94 percent!
        %
        %dname            = 'proc_coil_mlclass_unequal';
        %dname            = 'proc_coil_mlclassuneq_consec';
        load([datasetpath, dname,'.mat'],'data');
        data.datasetName = dname;
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 200; 
   case cnstDefs.COILPRRANGE
        datasetpath      = [datasetmain, '/coil/coil-20-proc/'];
        coilpr_ind         = datasetId - cnstDefs.COILPRSTART + 1; 
        fl               = cnstDefs.coilprfirst_label(coilpr_ind);
        sl               = cnstDefs.coilprsecond_label(coilpr_ind);
        dname            = ['proc_coil_b_', sprintf('%d',fl),'_', sprintf('%d',sl)];
        load([datasetpath, dname,'.mat'],'data_b');
        data             = data_b;
        data.datasetName = dname;
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 50; 
        data.Y           = data.Y';    
    case cnstDefs.COIL100MLCL
        datasetpath      = [datasetmain, '/coil-100/'];
        dname            = 'coil100_mlclass';
        load([datasetpath, dname,'.mat'],'data');
 
        data.datasetName = dname;
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 50; 
        index            = 1:100;
        minoneindex      = (-1).^index;
        Ylabel           = 2*unique(data.label).*minoneindex;
        data.Y           = Ylabel(data.label);
        data.binary      = false;
        n_lab            = 5;
        %label_list       = randi(100, n_lab, 1); %31    72    68    27    85    50    41    70    50    43    40    15    81    11     3    58    73    47    94 51
        %label_list       = [31    72    68    27    85    50    41    70    50    43    40    15    81    11     3    58    73    47    94 51];
        label_list       = [ 60    86    25    48    76];
        labsel           = Ylabel(sort(label_list));
        [data]           = samplebylabel(data, labsel);
    case cnstDefs.ARTICLESENCL   
        datasetpath      = [datasetmain, '/SentenceCorpus\SentenceCorpus\labeled_articles/'];
        dname            = 'ArticleSentence';
        load([datasetpath, dname,'.mat'],'data');
        data.binary      = false;
        data.datasetName = dname;
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 50; 
    case cnstDefs.CALVEHICLE
        datasetpath      = [datasetmain, '\Caltech101\101_ObjectCategories\CaltechSubsets\Vehicle/'];
        fname            = 'caltech_vehicle';
        load([datasetpath, fname,'.mat'],'data');
        dname            = 'CALTECHVEH';
        data.binary      = false;
        data.datasetName = dname;
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 50;
    case cnstDefs.CALANIM 
        datasetpath      = [datasetmain, '\Caltech101\101_ObjectCategories\CaltechSubsets\animals/'];
        fname            = 'calanimals';
        load([datasetpath, fname,'.mat'],'data');
        dname            = 'CALANIM';
 
        data.datasetName = dname;
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al        = 50;
        data.binary      = false;
    case cnstDefs.DBPEDIASMALLMLCL
        datasetpath   = [datasetmain, '\TextClassificationDatasets\dbpedia_csv\'];
        dname         = 'dbpediasmall';
        load([datasetpath, dname,'.mat'],'data');
        data.binary      = false;
        data.datasetName = dname;
        data.X           = double(data.X);
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 200;  
    case cnstDefs.AGNEWSSMALL
        datasetpath   = [datasetmain, '\TextClassificationDatasets\ag_news_csv\'];
        dname         = 'agnewssmall';
        load([datasetpath, dname,'.mat'],'data');
        data.binary      = false;
        data.datasetName = dname;
        data.X           = double(data.X);
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 200;
    case cnstDefs.YELPFULLSMALL
        datasetpath   = [datasetmain, '\TextClassificationDatasets\yelp_review_full_csv\'];
        dname         = 'yelpsmall';
        load([datasetpath, dname,'.mat'],'data');
        data.binary      = false;
        data.datasetName = dname;
        data.X           = double(data.X);
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 200;
    case cnstDefs.CRISIS_ALBERTFLD 
 
        crisisInd = crisisDInd;
        datasetpath      = [datasetmain, '..\CrisisDatasets\'];
        
        cname             = {'2013_Alberta_floods' , '2012_Colorado_wildfires', '2012_Costa_Rica_earthquake', ...
                             '2012_Guatemala_earthquake', '2012_Italy_earthquakes', '2012_Philipinnes_floods', ...
                             '2012_Typhoon_Pablo', '2013_Australia_bushfire', '2013_Bohol_earthquake', ...
                             '2013_Boston_bombings', '2013_Brazil_nightclub_fire', '2013_Colorado_floods',...
                             '2013_Glasgow_helicopter_crash', '2013_LA_airport_shootings', '2013_Lac_Megantic_train_crash',...
                             '2013_Manila_floods', '2013_NY_train_crash', '2013_Queensland_floods', ...
                             '2013_Russia_meteor', '2013_Sardinia_floods', '2013_Savar_building_collapse', ...
                             '2013_Singapore_haze', '2013_Spain_train_crash', '2013_Typhoon_Yolanda', ...
                             '2013_West_Texas_explosion', '2012_Venezuela_refinery'}; 
        fname            = cname{crisisInd};
        dname            = fname;
        load([datasetpath, fname,'-big.mat'],'data');
        data.binary      = false;
        data.datasetName = dname;
        data.X           = double(data.X);
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al        = 1000;   
        data.divide_func = @divide_data_crisis;
        %data.data_comp_kernel = @comp_twolevel_kernel;
        data.data_comp_kernel = @comp_simple_kernel_general;
        %data.data_comp_kernel = @comp_threeway_kernel;
        data.split_data_func  = @divide_crisis_data_by_Fid;
        data.summarizedinfo = @summarizedataset_crisis;
    case cnstDefs.CIFAR100CELL
        datasetpath   = [datasetmain, '\CIFAR100\\cifar-100-matlab\'];
        dID           = datasetId - cnstDefs.CIFAR100AQUATICMAMMALS + 1;
        dname         = {'AQUATICMAMMALS', 'ANIMALS', 'FLOWERS', 'FOODCONT', 'FRUITVEGET', 'HOUSEHOLD', 'INSECTS', 'LARGECARNIVOR', 'LARGEOMNIVORE',...
                         'MAMMALS', 'MANMADEOUTDOOR', 'NATURALSCENCE', 'NONINSECTSINVERT', 'PEOPLE', 'REPTILES', 'TREES', 'VEHICLES'};
                     
        fname         = {'mammals cifar100', 'Animals cifar100', 'flowers cifar100', 'food containers cifar100', 'fruit and vegetables cifar100',...
                          'household cifar100', 'insects cifar100', 'large carnivores cifar100', 'large omnivores and herbivores cifar100', ...
                          'mammals cifar100', 'man-made outdoor things cifar100', 'natural outdoor scenes cifar100', 'non-insect invertebrates cifar100',...
                          'people cifar100', 'reptiles cifar100', 'Trees cifar100', 'vehicles cifar100'} ;
        load([datasetpath, fname{dID},'.mat'],'data');
        data.binary      = false;
        data.datasetName = dname{dID};
        data.X           = double(data.X);
        data.noisy       = false(1,data.n);
        data.labelnoise  = false(1,data.n);
        data.d           = size(data.X, 1);
        data.n_al                = 200;
        data.binary      = false;
    otherwise 
        assert(false, 'Dataset Id is invalid');
end
data.LB_mapping_func     = LB_mapfunc_list{LB_mapping_method};
data.LB_settings         = LB_settings_list{LB_mapping_method}; 
[data          ]         = data.LB_mapping_func(data, data.LB_settings);

[data.learningparams_init ] = compute_learningparams(data, datasetId, gamma_ratio, gamma_o_ratio, lambda_o_ratio,kernel_func);
if isfield(data.learningparams_init, 'L2_lp_init') && data.isTwoLevel
    data.learningparams_init_L2 = data.learningparams_init.L2_lp_init;
end
data.dname       = data.datasetName;
data.datasetName = dname; 
% strcat(data.datasetName, sprintf('LAM=%5.3e-GAM=%5.3e-GAMIS=%5.3e-VARRHO=%3.2f', ...
%                                                     data.learningparams_init.lambda,...     
%                                                     data.learningparams_init.KOptions.gamma,...
%                                                     data.learningparams_init.KOptions.gamma_is, ...
%                                                     data.learningparams_init.varrho));
                                                               
data.DS_subsampling_func = DS_subsampling_list{subdataset_method};
data.DS_settings         = DS_settings{subdataset_method};
data.TR_sampling_func    = TR_subsampling_list{TR_subdataset_method};
data.TR_settings         = TR_settings{TR_subdataset_method};
data.UN_sampling_func    = UN_labeled_subset_query_list{UN_subsampling_method};
data.UN_settings         = UN_settings{UN_subsampling_method};
if profile.LOAD_SAVE_SVM_PARAMS % if it not exist compute it and store it. 
    fname = get_SVMparamsfilename(data_dir, datasetName);
    if exist(fname,'file') && ~recompute
       load(fname,'learning_params');
    else
       Classification_exp_param = set_classification_experiment('SVM' ,@SVMtrain ,@SVMtester, @SVM_learning_list_maker);                             
       temp_profile = profile;
       temp_profile.showdata    = false;
       temp_profile.CV_search_randomselect = false;
       learning_params          = cross_validation(Classification_exp_param, data, temp_profile, learning_params_init);  
       save(fname,'learning_params');
    end
end
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
comp_mean = true;
if comp_mean
    data =  comp_mean_vec_dist(data);
    data.has_mean_vec = true;
end
data.has_mean_vec = comp_mean;
if ~isfield(data, 'data_comp_kernel')
    data.data_comp_kernel = @comp_simple_kernel_general;
end
end
function data = load_USPDSDigits(datasetmain, digit1, digit2, samplesize, gensize,gensize_var, recomp) 
    dg1       = int2str(digit1);
    dg2       = int2str(digit2);
    datasetName = ['USPDSDIST', dg1,' ', dg2'] ;
    mutoydata = ['\USPSDIST',dg1, dg2,'.mat'];
    filename  = strcat(strcat(datasetmain,'\uspsdist\'), mutoydata);
    if ~recomp && exist(filename, 'file')
        load(filename, 'data');
    else
        s_avg   = [1;1];  vars    = 0.1;
        d_avg   = [0;0];  vard    = 5;
        theta   = 0;      vartheta= 3.14;
        [data]  = get_usps_virtualexamples(digit1, digit2, samplesize, gensize,gensize_var, s_avg, vars, d_avg, vard, theta, vartheta);
        save(filename, 'data');
    end
    data.noisy       = false(1,data.n);
    data.labelnoise  = false(1,data.n);                 
    data.datasetName = datasetName;
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
function [learningparams_init ] = compute_learningparams(data, datasetId, gamma_ratio, gamma_o_ratio, lambda_o_ratio,kernel_func)
global cnstDefs

lambda     = 2e-4;
lambda_AL  = lambda;
lambda_o   = lambda;
gamma_o    = 0.05;
%gamma_is   = 0.005;
sigma_is   = median_heurisitc(data.X);
varrho     = 0.1;
sigma_likelihood = 1;
BME_thau = 1;
L2_lp_init = [];
switch(datasetId)
    case {cnstDefs.DIST_SYNTH_MUGAMMA, cnstDefs.DIST_DATASET_GAUSSIAN_ONEDIM,...
          cnstDefs.DIST_SYNTH_MUANDETTOY1, cnstDefs.DIST_SYNTH_MUANDETTOY2,cnstDefs.DIST_SYNTH_MUANDETTOY3  }
        gamma_is   = 1/sigma_is^2;
        kernel_func= @rbf_expectedkernel;
        switch(datasetId)
            case cnstDefs.DIST_SYNTH_MUGAMMA
                 lambda     = 2e-1;
                 kernel_func= @rbf_expectedkernel;
                 %kernel_func= @rbf_twolevel;
                 gamma_is   = 1/sigma_is^2;
            case cnstDefs.DIST_DATASET_GAUSSIAN_ONEDIM   
                
            case cnstDefs.DIST_SYNTH_MUANDETTOY1 % this is a small dataset consist of 10 distributions

            case cnstDefs.DIST_SYNTH_MUANDETTOY2 % this is bigg
                 BME_thau = 1;
            case cnstDefs.DIST_SYNTH_MUANDETTOY3 % this is bigg
                 BME_thau = 0.1;
        end
        gamma      = 0.1;%10;
    case {cnstDefs.USPSDIST_6_9, cnstDefs.USPSDIST_3_8, cnstDefs.USPSDIST_1_8, cnstDefs.USPSDIST_3_4}
        lambda     = 2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
    case cnstDefs.MUSK1
        lambda     = 2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 4/sigma_is^2;%5
        gamma      = 1;
        %kernel_func= @rbf_twolevel;%rbf_expectedkernel;%@rbf_twolevel;
        kernel_func= @rbf_expectedkernel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
    case cnstDefs.MUSK2
        lambda     = 2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 10/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;    
     case cnstDefs.NEWSGROUPRANGE
        lambda     = 2e-5;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 1;
        lambda_AL  = 2e-5;
    case cnstDefs.NEWSGROUPONEVSRESTRANGE 
        lambda     = 2e-5;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 1;
        lambda_AL  = 2e-5;
    case cnstDefs.REUTERSRANGE
        lambda     = 2e-5;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 1;
        lambda_AL  = 2e-5;
   case cnstDefs.REUTERSMLCL
        lambda     = 2e-5;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 2/sigma_is^2;%5
        gamma      = 0.1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 10;
        lambda_AL  = 1;%2e-5;
       
   case cnstDefs.REUTERSONEVSRESTRANGE 
        lambda     = 2e-5;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 1;
        lambda_AL  = 2e-5;
    case cnstDefs.YELPPOLARITY
        lambda     = 2e-5;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 1;
        lambda_AL  = 2e-5;
    case cnstDefs.YELPSENT
        lambda     = 2e-1;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 2;
        lambda_AL  = lambda;
    case cnstDefs.IMDBPOLARITY
        lambda     = 2e-1;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 2;
        lambda_AL  = lambda;
    case cnstDefs.AMAZONPOLARITY     
        lambda     = 2e-1;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 2;
        lambda_AL  = lambda;
    case cnstDefs.CIFARRANGE     
        lambda     = 2e-1;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 2;
        lambda_AL  = lambda;    
    case cnstDefs.CIFARMLCL     
        lambda     = 2e-7;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 0.1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 2;
        lambda_AL  = lambda;    
     case cnstDefs.STL10MLCL
        lambda     = 2e-1;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 0.1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 2;
        lambda_AL  = lambda;
    case cnstDefs.COILUNRANGE
        lambda     = 2e-1;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 2;
        lambda_AL  = lambda;
    case cnstDefs.COILUNMLCL
        lambda     = 2e-3;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 0.1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 2;
        lambda_AL  = lambda;
    case cnstDefs.COILPRRANGE
        lambda     = 2e-1;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 0.1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 2;
        lambda_AL  = lambda;    
    case cnstDefs.COIL20MLCL
        lambda     = 2e-7;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 0.3;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 2;
        lambda_AL  = lambda;
    case cnstDefs.COIL100MLCL
        lambda     = 2e-1;%2e-5;1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 2/sigma_is^2;%
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 0.05;
        lambda_AL  = 1;%lambda;
    case cnstDefs.ARTICLESENCL   
        lambda     = 2e-5;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 2/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 5;
        lambda_AL  = 1;%2e-5;
    case cnstDefs.CALVEHICLE    
        lambda     = 2e-5;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 2/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 1;
        lambda_AL  = 0.1;%2e-5;     
    case cnstDefs.CALANIM    
        lambda     = 2e-6;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 1;
        lambda_AL  = 0.1;%2e-5;        
    case cnstDefs.DBPEDIASMALLMLCL
        lambda     = 2e-6;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 1;
        lambda_AL  = 0.1;%2e-5;
    case cnstDefs.AGNEWSSMALL
        lambda     = 2e-6;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 1;
        lambda_AL  = 0.1;%2e-5;    
    case cnstDefs.YELPFULLSMALL
        lambda     = 2e-6;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 1;
        lambda_AL  = 0.1;%2e-5;     
    case cnstDefs.CRISIS_ALBERTFLD 
        lambda     = 2e-6;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 1/sigma_is^2;%5
        gamma      = 1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        kernel_func_L2 = @myrbf_kernel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 10;
        BME_thau   = 1;
        lambda_AL  = 0.1;%2e-5;   
        lambdaL2   = 0.1;
        
        if data.isTwoLevel 
           L2_lp_init = learning_settings(data.n, 'gamma'      , gamma_is   , 'gamma_is', gamma_is, 'gamma_o', gamma_o,...
                                                  'kernel_func', kernel_func_L2, 'varrho', varrho, 'BME_thau' ,BME_thau,  ...
                                                  'lambda'     , lambdaL2     , 'lambda_AL', lambda_AL, 'lambda_o', lambda_o,'sigma_likelihood', sigma_likelihood, ...
                                                  'data_noisy' , data.noisy , 'data_labelnoise', data.labelnoise); 
        end
    case cnstDefs.CIFAR100FLOWERS
%         {cnstDefs.CIFAR100AQUATICMAMMALS, cnstDefs.CIFAR100ANIMALS, , cnstDefs.CIFAR100FOODCONTANIERS,...
%           cnstDefs.CIFAR100FRUITVEG, cnstDefs.CIFAR100HOUSEHOLD, cnstDefs.CIFAR100INSECTS, cnstDefs.CIFAR100LARGECARNIVOR,...
%           cnstDefs.CIFAR100LARGEOMNIVORE, cnstDefs.CIFAR100MAMMALS, cnstDefs.CIFAR100MANMADEOUTDOOR, cnstDefs.CIFAR100NATURALSCENCE, cnstDefs.CIFAR100NONINSECTSINVERT 
%           cnstDefs.CIFAR100PEOPLE, cnstDefs.CIFAR100REPTILES, cnstDefs.CIFAR100TREES, cnstDefs.CIFAR100VEHICLES}
        lambda     = 2e-5;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 0.5/sigma_is^2;%5
        gamma      = 0.1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 0.1;
        BME_thau   = 0.1;
        lambda_AL  = 2e-3;
    case cnstDefs.CIFAR100CELL
%         {cnstDefs.CIFAR100AQUATICMAMMALS, cnstDefs.CIFAR100ANIMALS, cnstDefs.CIFAR100FLOWERS, cnstDefs.CIFAR100FOODCONTANIERS,...
%           cnstDefs.CIFAR100FRUITVEG, cnstDefs.CIFAR100HOUSEHOLD, cnstDefs.CIFAR100INSECTS, cnstDefs.CIFAR100LARGECARNIVOR,...
%           cnstDefs.CIFAR100LARGEOMNIVORE, cnstDefs.CIFAR100MAMMALS, cnstDefs.CIFAR100MANMADEOUTDOOR, cnstDefs.CIFAR100NATURALSCENCE, cnstDefs.CIFAR100NONINSECTSINVERT 
%           cnstDefs.CIFAR100PEOPLE, cnstDefs.CIFAR100REPTILES, cnstDefs.CIFAR100TREES, cnstDefs.CIFAR100VEHICLES}
        lambda     = 2e-5;%2e-1;%1;%2e-5;%1;%1;%2e-1;
        sigma_likelihood = 1;
        lambda_o   = lambda;
        gamma_is   = 0.5/sigma_is^2;%5
        gamma      = 0.1;
        kernel_func= @rbf_expectedkernel;%@rbf_twolevel;
        r_kernel_func = @r_rbf_expectedkernel;
        varrho     = 0.1;
        BME_thau   = 0.1;
        lambda_AL  = 2e-1;
end
  

if ~data.isTwoLevel 
    [learningparams_init ]  = learning_settings(data.n, 'gamma'     , gamma, 'gamma_is', gamma_is, 'gamma_o', gamma_o,...
                                                     'kernel_func',kernel_func, 'varrho', varrho, 'BME_thau' ,BME_thau,  ...
                                                     'lambda'    , lambda, 'lambda_AL', lambda_AL, 'lambda_o', lambda_o,'sigma_likelihood', sigma_likelihood, ...
                                                     'data_noisy', data.noisy, 'data_labelnoise', data.labelnoise);
    learningparams_init.isTwoLevel = 0;
else
    [learningparams_init ]  = learning_settings(data.n, 'gamma'     , gamma, 'gamma_is', gamma_is, 'gamma_o', gamma_o,...
                                                     'kernel_func',kernel_func, 'varrho', varrho, 'BME_thau' ,BME_thau,  ...
                                                     'lambda'    , lambda, 'lambda_AL', lambda_AL, 'lambda_o', lambda_o,'sigma_likelihood', sigma_likelihood, ...
                                                     'data_noisy', data.noisy, 'data_labelnoise', data.labelnoise);
    learningparams_init.C_g        = 2;
    learningparams_init.C_v        = 1;
    learningparams_init.Delta      = 10;
    learningparams_init.isTwoLevel = 1;
    learningparams_init.L2_lp_init = L2_lp_init;                                                
end
end
function [data] = get_save_data(filename, data_generator, par, varsize, avgsize, sigsize, recomp, gamma_ratio, gamma_o_ratio, lambda_o_ratio)
if ~recomp && exist(filename, 'file')
    load(filename, 'data');
else
   data      = data_generator(par,varsize, avgsize, sigsize);
%    data.gamma_is = median(median(pdist2(data.X',data.X')));
%    data.dm = distance_matrix(data, data, data.gamma_is, true);
%    [data.lambda, data.lambda_o, data.gamma, data.gamma_o ] = compute_DISTgamma(data, gamma_ratio, gamma_o_ratio, lambda_o_ratio);
%    data.K  = comp_KernelEmb(data.dm, data.gamma);
   save(filename, 'data');
end
data.noisy       = false(1,data.n);
data.labelnoise  = false(1,data.n);
end