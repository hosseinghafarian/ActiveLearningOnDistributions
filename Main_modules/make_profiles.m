function [profile ] = make_profiles(active_profile_id, synthmode, distributional)
global cnstDefs
if ~distributional
   if ~synthmode
      [showOptions]  = set_showoptions();
      [dataset_list] = get_datasetlist(cnstDefs.INPUTSPACEDATASETS);%get_all_datasets_in_alist();
   else
      [showOptions] = set_showoptions('showData',true); 
      [dataset_list] = get_datasetlist(cnstDefs.SYNTHDATASETS);%get_all_synthesisdatasets_in_alist();
   end
else
   if ~synthmode
      [showOptions]  = set_showoptions();
      [dataset_list] = get_datasetlist(cnstDefs.DISTSPACEDATASETS);%get_all_DISTdatasets_in_alist();
   else
      [showOptions] = set_showoptions('showData',false); 
      [dataset_list] = get_datasetlist(cnstDefs.DISTSYNTHDATASETS);%get_all_DISTsynthesisdatasets_in_alist();
   end
end
switch active_profile_id
    case cnstDefs.SYNTHDATA
       profile = set_profile(dataset_list, showOptions,'synthesis'    , true, 'transductive', true, 'showdata', true,...
                                                         'testratio'       , 1,    'trainratio',1, 'DSsubsampling_id', 1 ,...
                                                         'CVsubsampling_id', 1,'UNsubsampling_id'      ,1, 'TRsubsampling_id'      ,1,...
                                                         'LBmaping_id'     , 1,'labelnoisepercent'     ,0, 'outlierpercent'        ,0,...
                                                         'kfold',1 ,'CV_search_randomselect',false       ,'LOAD_SAVE_SVM_PARAMS'  ,false);
    case cnstDefs.SYNTHNOACTIVELEARNING
       profile =  set_profile(dataset_list, showOptions,'synthesis'    , true, 'transductive'      , true, 'showdata'   , true,...
                                                         'testratio'       , 1, 'trainratio'            ,1, 'do_al'       ,false, 'DSsubsampling_id', 1 ,...
                                                         'CVsubsampling_id', 1, 'UNsubsampling_id'      ,1, 'TRsubsampling_id',1,...
                                                         'LBmaping_id'     , 1, 'labelnoisepercent'     ,0, 'outlierpercent'  ,0,...
                                                         'kfold',1 ,'CV_search_randomselect',false        ,'LOAD_SAVE_SVM_PARAMS'  ,false);
    case cnstDefs.CV_CLASSIFICATION
        profile = set_profile(dataset_list, showOptions, 'do_al', false, 'do_cl', true, 'Full_CV_search', true, 'kfold', 10, 'distributional', distributional);
    case cnstDefs.RAPIDCV_CLASSIFICATION
        profile = set_profile(dataset_list, showOptions, 'do_al', false, 'Full_CV_search', false);
    case cnstDefs.FASTCV_CLASSIFICATION
        profile = set_profile(dataset_list, showOptions, 'do_al', false, 'do_cl', true, 'Full_CV_search', true, 'kfold', 10, 'distributional', distributional);
    case cnstDefs.FASTSMALLCV_CLASSIFICATION
        profile = set_profile(dataset_list, showOptions, 'do_al', false, 'Full_CV_search', false, 'kfold', 2,...
                                                          'CV_search_notlessthan' ,3, 'CV_search_notmorethan' ,20 , 'distributional', distributional);
    case cnstDefs.TINYCV_CLASSIFICATION
        profile = set_profile(dataset_list, showOptions, 'do_al', false, 'Full_CV_search', false, 'kfold', 2,...
                                                          'CV_search_notlessthan' ,2, 'CV_search_notmorethan' ,3 );                                                                        
    case cnstDefs.LOADPARAMS_CLASSIFICATION
        profile = set_profile(dataset_list, showOptions, 'do_al', false, 'load_lp_from_file',true);
    case cnstDefs.CV_CLASSIFICATION_ACTIVE
        profile =  set_profile(dataset_list, showOptions);
    case cnstDefs.ACTIVELEARNING
        profile =  set_profile(dataset_list, showOptions, 'do_cv',false,'do_cl',false,'do_al',true,'trainratio',0.8,...
                                                          'testratio',0.2,'LOAD_SAVE_SVM_PARAMS'  ,false, 'distributional',distributional );
    case cnstDefs.ACTIVELEARNING_SMALLDATASET
        profile =  set_profile(dataset_list, showOptions, 'do_cv',false,'do_cl',false,'do_al',true,'trainratio',0.5,...
                                                          'testratio',0.5, 'DSsubsampling_id', 2, 'DSsubsampling_per', 20,'TRadjustsizemethod_id' ,2, 'TRsample_larger_than'  ,40 );
    %distributional
    case cnstDefs.SYNTHDISTDATA
        profile = set_profile(dataset_list, showOptions,'distributional', true, 'synthesis'    , true, 'transductive', false, 'showdata', false,...
                                                        'do_cl', true,...
                                                         'testratio'       , 0.8,    'trainratio',0.2, 'DSsubsampling_id', 1 ,...
                                                         'CVsubsampling_id', 1,'UNsubsampling_id'      ,1, 'TRsubsampling_id'      ,1,...
                                                         'LBmaping_id'     , 1,'labelnoisepercent'     ,0, 'outlierpercent'        ,0,...
                                                         'kfold',1 ,'CV_search_randomselect',false       ,'LOAD_SAVE_SVM_PARAMS'  ,false);
    case cnstDefs.SYNTHSMMPAPERDISTDATA
        dataset_list = [ cnstDefs.DIST_SYNTH_MUANDETTOY2, cnstDefs.DIST_SYNTH_MUANDETTOY1];
        profile = set_profile(dataset_list, showOptions,'distributional', true, 'synthesis'    , true, 'transductive', false, 'showdata', false,...
                                                        'do_cl', true, 'do_cv', true, 'do_al', false, ...
                                                        'testratio'       , 2/12, 'trainratio',10/12, 'DSsubsampling_id', 1 ,...
                                                        'CVsubsampling_id', 1,'UNsubsampling_id'      ,1, 'TRsubsampling_id'      ,1,...
                                                        'LBmaping_id'     , 1,'labelnoisepercent'     ,0, 'outlierpercent'        ,0,...
                                                        'kfold',10 ,'CV_search_randomselect',false       ,'LOAD_SAVE_SVM_PARAMS'  ,false);    
    case cnstDefs.SYNTHSMMPAPERDISTDATA_ALEXP
        
        profile = set_profile(dataset_list, showOptions,'distributional', true, 'synthesis'    , true, 'transductive', false, 'showdata', false,...
                                                        'do_cl', false, 'do_cv', false, 'do_al', true, ...
                                                        'testratio'       , 2/12, 'trainratio',10/12, 'DSsubsampling_id', 1 ,...
                                                        'CVsubsampling_id', 1,'UNsubsampling_id'      ,1, 'TRsubsampling_id'      ,1,...
                                                        'LBmaping_id'     , 1,'labelnoisepercent'     ,0, 'outlierpercent'        ,0,...
                                                        'kfold',10 ,'CV_search_randomselect',false       ,'LOAD_SAVE_SVM_PARAMS'  ,false);                                                                             
end
end
function [profile] = set_profile(dataset_list, showOptions, varargin)
global cnstDefs
params = inputParser;
    params.addParameter('synthesis'             ,false           ,@(x) islogical(x));
    params.addParameter('transductive'          ,false           ,@(x) islogical(x)); 
    params.addParameter('dontshuffle'           ,false           ,@(x) islogical(x)); 
    params.addParameter('distributional'        ,false           ,@(x) islogical(x));
    params.addParameter('showdata'              ,false           ,@(x) islogical(x));
    params.addParameter('testratio'             ,0.2             ,@(x) isscalar(x) & x>0);
    params.addParameter('trainratio'            ,0.8             ,@(x) isscalar(x) & x>0);
    params.addParameter('do_cv'                 ,true            ,@(x) islogical(x));
    params.addParameter('do_cvforobtainlp'      ,false           ,@(x) islogical(x));
    params.addParameter('do_cl'                 ,false           ,@(x) islogical(x));
    params.addParameter('do_al'                 ,true            ,@(x) islogical(x));
    %params.addParamValue('solver'              ,'sdpt3'         ,@(x) ischar(x));
    params.addParameter('solver'                ,'mosek'         ,@(x) ischar(x));
    params.addParameter('solververbose'         ,cnstDefs.solver_verbose   ,@(x) islogical(x));
    params.addParameter('save_lp_to_file'       ,false            ,@(x) islogical(x));
    params.addParameter('load_lp_from_file'     ,false           ,@(x) islogical(x));
    params.addParameter('save_lp_filename'      ,'learning_params_file'  ,@(x) ischar(x));
    params.addParameter('Full_CV_search'        ,true            ,@(x) islogical(x));
    params.addParameter('LOAD_SAVE_SVM_PARAMS'  ,false            ,@(x) islogical(x));
    params.addParameter('CV_search_percent'     ,10              ,@(x) isscalar(x) & x>=0 );
    params.addParameter('CV_search_notlessthan' ,5               ,@(x) isscalar(x) & x>=0 );
    params.addParameter('CV_search_notmorethan' ,100             ,@(x) isscalar(x) & x>=0 );
    params.addParameter('CV_search_randomselect',false           ,@(x) islogical(x));
    params.addParameter('CV_search_rndsel_per'  ,20              ,@(x) isscalara(x) &x > 0 & x<=100);
    params.addParameter('kfold'                 ,10              ,@(x) isscalar(x) & x>=0 );
    params.addParameter('DSsubsampling_id'      ,1               ,@(x) isscalar(x) & x>= 1 & x<=3);
    params.addParameter('DSsubsampling_per'     ,80              ,@(x) isscalar(x) & x>= 0 & x<=100);
    params.addParameter('CVsubsampling_id'      ,1               ,@(x) isscalar(x) & x>= 1 & x<=3);
    params.addParameter('CVsubsampling_per'     ,100              ,@(x) isscalar(x) & x>= 0 & x<=100);
    params.addParameter('UNsubsampling_id'      ,2               ,@(x) isscalar(x) & x>= 1 & x<=3);
    params.addParameter('UNsubsampling_per'     ,80              ,@(x) isscalar(x) & x>= 0 & x<=100);
    params.addParameter('TRsubsampling_id'      ,1               ,@(x) isscalar(x) & x>= 1 & x<=3);
    params.addParameter('TRsubsampling_per'     ,80              ,@(x) isscalar(x) & x>= 0 & x<=100);
    params.addParameter('TRadjustsizemethod_id' ,2               ,@(x) isscalar(x) & x>= 1 & x<=3);
    params.addParameter('TRsample_larger_than'  ,40              ,@(x) isscalar(x) & x> 0);
    params.addParameter('LBmaping_id'           ,1               ,@(x) isscalar(x) & x>= 1 & x<=4);
    params.addParameter('labelnoisepercent'     ,25              ,@(x) isscalar(x) & x>=0 );
    params.addParameter('outlierpercent'        ,25              ,@(x) isscalar(x) & x>=0 );
    params.addParameter('unbalance_LB_tolerance',5               ,@(x) isscalar(x) & x>=0 );
    
    params.parse(varargin{:});
    profile               = params.Results;
    profile.dataset_list  = dataset_list;
    profile.showOptions   = setshowOptions();
    profile.showOptions.showData = profile.showdata;  
    profile.Options.Transductive = profile.transductive;
    profile.Options.isDistData   = false;
    profile.showOptions          = showOptions;
end
function [showoptions] = set_showoptions(varargin)
    params = inputParser;
    params.addParameter('synthesis'             ,false           ,@(x) islogical(x));
    params.addParameter('transductive'          ,false           ,@(x) islogical(x));
    params.addParameter('showData'              ,false           ,@(x) islogical(x));
    params.addParameter('showContour'           ,false           ,@(x) islogical(x));
    params.addParameter('showIndex'             ,true            ,@(x) islogical(x));
    params.addParameter('isUpdate'              ,false           ,@(x) islogical(x));
    params.addParameter('clear'                 ,false           ,@(x) islogical(x));
    params.addParameter('StopforKey'            ,false           ,@(x) islogical(x));
    params.addParameter('figName'               ,'Samples'       ,@(x) ischar(x));     
    params.parse(varargin{:});
    showoptions  = params.Results;
end