function globaldefinitions(setup_path)
global cnstDefs
%global settings 
%profile number settings 
cnstDefs.SYNTHDATA                 = 1;
cnstDefs.CV_CLASSIFICATION         = 2;
cnstDefs.RAPIDCV_CLASSIFICATION    = 3;
cnstDefs.FASTCV_CLASSIFICATION     = 4;
cnstDefs.LOADPARAMS_CLASSIFICATION = 5;
cnstDefs.CV_CLASSIFICATION_ACTIVE  = 6;
cnstDefs.FASTSMALLCV_CLASSIFICATION= 7;
cnstDefs.TINYCV_CLASSIFICATION     = 4;
cnstDefs.ACTIVELEARNING            = 8;
cnstDefs.ACTIVELEARNING_SMALLDATASET = 9;
cnstDefs.SYNTHNOACTIVELEARNING       = 10;
cnstDefs.SYNTHDISTDATA               = 11;
cnstDefs.SYNTHSMMPAPERDISTDATA       = 12; % A profile for replicating experiments of SMM paper by muandet
cnstDefs.SYNTHSMMPAPERDISTDATA_ALEXP = 13; % A profile for active learning experiments using the above data

% paths
cnstDefs.main_path       = setup_path;
cnstDefs.dataset_path    = '../Datasets/';
cnstDefs.exceptions_path = '\Exceptions\';
cnstDefs.result_path     = '\Results\';
cnstDefs.result_classification_path = '\Results\CL\';
cnstDefs.result_activelearning_path = '\Results\AL\';
cnstDefs.word2vecmodels_path = '\libs_ext\word2vec_matlab-master\';
cnstDefs.learningparams_path = '\Learningsettings\';%storing and loading results of learning_settings searches
% Output settings
cnstDefs.solver_verbose = false;
cnstDefs.solver         = 'mosek';
%
cnstDefs.NO_INS        = 0;
cnstDefs.SINGLE_INS    = 1;
cnstDefs.LABELED_INS   = 2;
cnstDefs.UNLABELED_INS = 3;
cnstDefs.EXTIND_INS    = 4;
cnstDefs.P_INS         = 5;% Constraints related to p. 
% constraints ID 
cnstDefs.CSTR_ID_SUMQ                   = 1;
cnstDefs.CSTR_ID_LASTEL                 = 2;
cnstDefs.CSTR_ID_DIAG_INITL_P           = 3;
cnstDefs.CSTR_ID_DIAG_P_Q_UNLAB_QUERY   = 4;
cnstDefs.CSTR_ID_DIAG_P_Q_UNLAB_NOQUERY = 5;
cnstDefs.CSTR_ID_DIAGQ_Q                = 6;
cnstDefs.CSTR_ID_NOISE_P_INITL          = 7;
cnstDefs.CSTR_ID_NOISE_P_UNLABELED      = 8;
cnstDefs.CSTR_ID_ABSV_DIAG_UNLABELED_PE = 9;
cnstDefs.CSTR_ID_ABSV_DIAG_UNLABELED_NE = 10;
cnstDefs.CSTR_ID_ABSV_VL_YL_WO          = 11;
cnstDefs.CSTR_ID_ABS_WO_P_PE            = 12;
cnstDefs.CSTR_ID_ABS_WO_P_NE            = 13;
% Active Learning Method IDs
cnstDefs.ALMETHOD_ID_RANDSAMPLE         = 1;
cnstDefs.ALMETHOD_ID_QUIRE              = 2;
cnstDefs.ALMETHOD_ID_NOACTIVE           = 3;
cnstDefs.ALMETHOD_ID_UNCERTAINSAMPLE    = 4;
cnstDefs.ALMETHOD_ID_GUSSHARMONIC       = 5;
cnstDefs.ALMETHOD_ID_SOLVERROBUSTAL     = 6;
cnstDefs.ALMETHOD_ID_ALGORITHMROBUSTAL  = 7;
cnstDefs.ALMETHOD_ID_DIVERSEMULTITASK   = 8;
cnstDefs.ALMETHOD_ID_CRSEMISVM03        = 9;
cnstDefs.ALMETHOD_ID_DCONVXRELAX2       = 10;
cnstDefs.ALMETHOD_ID_DCVXRELAXPROXADMM  = 11;
cnstDefs.ALMETHOD_ID_MARGIN             = 12;
cnstDefs.ALMETHOD_ID_MAED               = 13;% manifold adaptive experiment design
cnstDefs.ALMETHOD_ID_MINMAX_SVMAL       = 14;
cnstDefs.ALMETHOD_ID_SOLVERRAL_ABS_H    = 15;
cnstDefs.ALMETHOD_ID_SLVRRAL_ABS_H_ABS_P= 16;
cnstDefs.ALPASSIVE_ID_SIMPLECOMPLEXCLS_1= 17;
cnstDefs.ALMETHOD_ID_SOLVERROBUSTALNORMONEPENALTY = 18;
cnstDefs.ALMETHOD_ID_MYPYCONDX          = 19;
cnstDefs.ALMETHOD_ID_SPECTRAL_E         = 20;
cnstDefs.ALMETHOD_ID_BAYESUNCERT        = 21;
cnstDefs.ALMETHOD_ID_SPECTL2BOOST       = 22;
cnstDefs.ALMETHOD_ID_SVDTRUNCATE        = 23;
cnstDefs.ALMETHOD_ID_FSIMPCOMPL2BOOST   = 24;
cnstDefs.ALMETHOD_ID_FSIMP3COMPL2BOOST  = 25;
cnstDefs.ALMETHOD_ID_QUIRECLUST         = 26; 
cnstDefs.ALMETHOD_ID_QUIRELAP           = 27; 
cnstDefs.ALMETHOD_ID_SPARSEPMAL         = 28;
cnstDefs.ALMETHOD_ID_ADAPTREGMARGIN     = 29;
cnstDefs.ALMETHOD_ID_ACTIVEBOOST        = 30;
cnstDefs.ALMETHOD_ID_L2BOOSTMINMAX      = 31;

cnstDefs.DISTALMETHOD_ID_RANDSAMPLE         = 101;
cnstDefs.DISTALMETHOD_ID_QUIRE              = 102;
cnstDefs.DISTALMETHOD_ID_NOACTIVE           = 103;
cnstDefs.DISTALMETHOD_ID_UNCERTAINSAMPLE    = 104;
cnstDefs.DISTALMETHOD_ID_GUSSHARMONIC       = 105;
cnstDefs.DISTALMETHOD_ID_SOLVERROBUSTAL     = 106;
cnstDefs.DISTALMETHOD_ID_ALGORITHMROBUSTAL  = 107;
cnstDefs.DISTALMETHOD_ID_DIVERSEMULTITASK   = 108;
cnstDefs.DISTALMETHOD_ID_CRSEMISVM03        = 109;
cnstDefs.DISTALMETHOD_ID_DCONVXRELAX2       = 110;
cnstDefs.DISTALMETHOD_ID_DCVXRELAXPROXADMM  = 111;
cnstDefs.DISTALMETHOD_ID_MARGIN             = 112;
cnstDefs.DISTALMETHOD_ID_MAED               = 113;% manifold adaptive experiment design
cnstDefs.DISTALMETHOD_ID_MINMAX_SVMAL       = 114;
cnstDefs.DISTALMETHOD_ID_SOLVERRAL_ABS_H    = 115;
cnstDefs.DISTALMETHOD_ID_SLVRRAL_ABS_H_ABS_P= 116;
cnstDefs.DISTALPASSIVE_ID_SIMPLECOMPLEXCLS_1= 117;
cnstDefs.DISTALMETHOD_ID_SOLVERROBUSTALNORMONEPENALTY = 118;
cnstDefs.HILBALMETHOD_ID_QUIRE              = 119;
cnstDefs.HILBALMETHOD_ID_BAYESUNCT          = 120;
cnstDefs.BKEAL_ID_BAYESUNCT                 = 121;
%Datasetlist Identifiers 
cnstDefs.LETTERDATASETS                 = 1;
cnstDefs.SSLDATASETS                    = 2;
cnstDefs.INPUTSPACEDATASETS             = 3;
cnstDefs.SYNTHDATASETS                  = 4;
cnstDefs.DISTSPACEDATASETS              = 5;
cnstDefs.DISTSYNTHDATASETS              = 6;
%Dataset Identifiers
cnstDefs.BREAST                         =  1;
cnstDefs.BUPALIVER                      =  2;
cnstDefs.ISOLET_1234                    =  3;
cnstDefs.ISOLET_5                       =  4;
cnstDefs.LETTER_EvsD                    =  5;
cnstDefs.LETTER_PvsD                    =  6;%6,7,8,9 :related to LETTER
cnstDefs.LETTER_EvsF                    =  7;
cnstDefs.LETTER_IvsJ                    =  8;
cnstDefs.LETTER_MvsN                    =  9;
cnstDefs.LETTER_UvsV                    = 10;
cnstDefs.MNIST                          = 11;
cnstDefs.VEHICLE                        = 12;
cnstDefs.IONOSHPERE                     = 13;
cnstDefs.SSLBOOK_DIGIT1                 = 14;
cnstDefs.SSLBOOK_USPS                   = 15;
cnstDefs.SSLBOOK_COIL2                  = 16;
cnstDefs.SSLBOOK_BCI                    = 17;
cnstDefs.SSLBOOK_G241C                  = 18;
cnstDefs.SSLBOOK_G241N                  = 19;
cnstDefs.SSLBOOK_COIL                   = 20;
cnstDefs.SSLBOOK_TEXT                   = 21;
cnstDefs.ADULT                          = 22;
cnstDefs.ECOLI                          = 23;
cnstDefs.GLASS                          = 24;
cnstDefs.HEART_STATLOG                  = 25;
cnstDefs.IMAGESEGMENT_STATLOG           = 26;
cnstDefs.PIMA_DIABETES                  = 27;
cnstDefs.SATELLITE                      = 28; 
cnstDefs.BREAST_WPDC                    = 29;
cnstDefs.SYNTH4OUTLIERORDERED                = 30;
cnstDefs.SYNTHSYNTH4OUTLIER_CORRECTLYLABELED = 31;
cnstDefs.SYNTHWITHOUTOUTLIERORDERED          = 32;
cnstDefs.SYNTHLINEARSVM                      = 33;
cnstDefs.SYNTHMORECOMPLEXTESTSIMPLEFUNC      = 34;
cnstDefs.SYNTHORDEREDOUTLIERLARGER           = 35;
cnstDefs.SYNTHORDEREDLITTLE_WITHOUTOUTLIER   = 36;
cnstDefs.SYNTHORDERED_DENSE_TWOOUTLIERS      = 37;
cnstDefs.SYNTHORDERED_DENSE_ONEOUTLIERS      = 38;
cnstDefs.SYNTHONEOUTLIER_ANOTHER             = 39;
cnstDefs.SYNTH2OUTLIER_FARAWAY               = 40;
cnstDefs.SYNTH4OUTLIER_INTHESAMEDIRECTION    = 41;
cnstDefs.SYNTH4OUTLIER_TWOINBOUNDRY          = 42;
cnstDefs.SYNTH4OUTLIER_TWOINBOUNDRY_MORELARGER = 43;
cnstDefs.SYNTH6OUTLIER_TWOINBOUNDRY_MORELARGER = 44;
cnstDefs.SYNTH1OUTLIER_INBOUNDRYSMALL          = 45;
cnstDefs.SYNTH1OUTLIER_INBOUNDRYSMALL_FURTHER  = 46;
cnstDefs.SYNTH1OUTLIER_CORRECTLYLABELED        = 47;
cnstDefs.SYNTH1OUTLIER_CORRECTLYLABELED_FURTHER  = 48;
cnstDefs.SYNTH1OUTLIER_CORRECTLYLABELED_FURTHER2 = 49;
cnstDefs.SYNTH6OUTLIER_NEAR                      = 50;
cnstDefs.SYNTHTESTSMO                            = 51;
cnstDefs.SYNTHLABELNOISE                         = 52;
cnstDefs.SYNTH2OUTLIER_INCORRECTLYLABELED        = 53;
cnstDefs.SYNTH3OUTLIER_CORRECTLYLABELED          = 54;
cnstDefs.SYNTH4OUTLIER_CORRECTLYLABELED          = 55;
cnstDefs.CODRNA                                  = 56;
cnstDefs.COLON                                   = 57;
cnstDefs.COVTYPE                                 = 58;
cnstDefs.FOURCLASS                               = 59;
cnstDefs.GERMANNUMER = 60;
cnstDefs.GISETTE     = 61;
cnstDefs.IJCNN1      = 62;
cnstDefs.LIVERDISORDER = 63;
cnstDefs.MADELON       = 64;
cnstDefs.MUSHROOM      = 65;
cnstDefs.NEWS20BINARY  = 66;
cnstDefs.PHISHING      = 67;
cnstDefs.REALSIM       = 68;
cnstDefs.SKINNOSKIN    = 69;
cnstDefs.sonar         = 70;
cnstDefs.svmguide1 = 71;
cnstDefs.svmguide3 = 72;
cnstDefs.w1a       = 73;
cnstDefs.w2a       = 74;
cnstDefs.w3a       = 75;
cnstDefs.w4a       = 76;
cnstDefs.w6a       = 77;
cnstDefs.w7a       = 78;
cnstDefs.w8a       = 79;

cnstDefs.a1a = 80;
cnstDefs.a2a = 81;
cnstDefs.a3a = 82;
cnstDefs.a4a = 83;
cnstDefs.a5a = 84;
cnstDefs.a6a = 85;
cnstDefs.a7a = 86;
cnstDefs.a8a = 87;
cnstDefs.a9a = 88;
cnstDefs.w5a       = 89;

cnstDefs.DIST_DATASET_GAUSSIAN_ONEDIM  = 101;
cnstDefs.DIST_SYNTH_MUGAMMA            = 102;
cnstDefs.DIST_SYNTH_MUANDETTOY1        = 103;
cnstDefs.DIST_SYNTH_MUANDETTOY2        = 104;
cnstDefs.DIST_SYNTH_MUANDETTOY3        = 105;
cnstDefs.USPSDIST_6_9                  = 130;
cnstDefs.USPSDIST_3_8                  = 131;
cnstDefs.USPSDIST_1_8                  = 132;
cnstDefs.USPSDIST_3_4                  = 133;
cnstDefs.MUSK1                         = 134;
cnstDefs.MUSK2                         = 135;
[cnstDefs.first_label, cnstDefs.second_label, newsgroup_num] = one_vs_one(1, 20);
cnstDefs.NEWSGROUPSTART                = 136;
cnstDefs.NEWSGROUPEND                  = cnstDefs.NEWSGROUPSTART+newsgroup_num-1;
cnstDefs.NEWSGROUPIDS                  = cnstDefs.NEWSGROUPSTART : cnstDefs.NEWSGROUPEND;
cnstDefs.NEWSGROUPRANGE                = num2cell(cnstDefs.NEWSGROUPIDS); 

cnstDefs.NEWSGROUPONEVSRESTSTART       = cnstDefs.NEWSGROUPEND + 1;
cnstDefs.NEWSGROUPONEVSRESTEND         = cnstDefs.NEWSGROUPONEVSRESTSTART + 20 - 1;
cnstDefs.NEWSGROUPONEVSRESTIDS         = cnstDefs.NEWSGROUPONEVSRESTSTART : cnstDefs.NEWSGROUPONEVSRESTEND;
cnstDefs.NEWSGROUPONEVSRESTRANGE       = num2cell(cnstDefs.NEWSGROUPONEVSRESTIDS); 

[cnstDefs.cifarfirst_label, cnstDefs.cifarsecond_label, cifar_num] = one_vs_one(0, 9);
cnstDefs.CIFARSTART                    = cnstDefs.NEWSGROUPONEVSRESTEND + 1;
cnstDefs.CIFAREND                      = cnstDefs.CIFARSTART+cifar_num;
cnstDefs.CIFARIDS                      = cnstDefs.CIFARSTART:cnstDefs.CIFARSTART+cifar_num-1;
cnstDefs.CIFARRANGE                    = num2cell(cnstDefs.CIFARIDS);

[cnstDefs.firreu_label, cnstDefs.secreu_label, reuters_num] = one_vs_one(1, 8);
cnstDefs.REUTERSSTART                  = cnstDefs.CIFAREND + 1;
cnstDefs.REUTERSEND                    = cnstDefs.REUTERSSTART+reuters_num-1;
cnstDefs.REUTERSIDS                    = cnstDefs.REUTERSSTART : cnstDefs.REUTERSEND;
cnstDefs.REUTERSRANGE                  = num2cell(cnstDefs.REUTERSIDS); 

cnstDefs.REUTERSONEVSRESTSTART         = cnstDefs.REUTERSEND + 1;
cnstDefs.REUTERSONEVSRESTEND           = cnstDefs.REUTERSONEVSRESTSTART + 8 - 1;
cnstDefs.REUTERSONEVSRESTIDS           = cnstDefs.REUTERSONEVSRESTSTART : cnstDefs.REUTERSONEVSRESTEND;
cnstDefs.REUTERSONEVSRESTRANGE         = num2cell(cnstDefs.REUTERSONEVSRESTIDS);
cnstDefs.YELPPOLARITY                  = cnstDefs.REUTERSONEVSRESTEND + 1;
cnstDefs.YELPSENT                      = cnstDefs.YELPPOLARITY + 1;
cnstDefs.IMDBPOLARITY                  = cnstDefs.YELPSENT + 1;
cnstDefs.AMAZONPOLARITY                = cnstDefs.IMDBPOLARITY + 1;
end