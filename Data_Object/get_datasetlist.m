function [dataset_list] = get_datasetlist(datasetlist_ID)
global cnstDefs
switch(datasetlist_ID)
    case cnstDefs.INPUTSPACEDATASETS
         %dataset_list = [get_all_datasets_in_alist(),get_letterds(),get_SSLds()];
         %dataset_list = get_letterds();
         %dataset_list = get_SSLds();
         dataset_list = get_all_datasets_in_alist();
    case cnstDefs.SYNTHDATASETS
         [dataset_list] = get_all_synthesisdatasets_in_alist();
    case cnstDefs.DISTSPACEDATASETS 
         [dataset_list] = get_all_DISTdatasets_in_alist();
    case cnstDefs.DISTSYNTHDATASETS
         [dataset_list] = get_all_DISTsynthesisdatasets_in_alist();
    case cnstDefs.LETTERDATASETS
         [dataset_list] = get_letterds();
    case cnstDefs.SSLDATASETS
         [dataset_list] = get_SSLds();
end

end
function [dataset_list] = get_all_datasets_in_alist()
global cnstDefs
dataset_testok = [ cnstDefs.LIVERDISORDER, cnstDefs.MUSHROOM,...
                   cnstDefs.PHISHING, cnstDefs.IONOSHPERE];
             
end

function [dataset_list] = get_letterds()
global cnstDefs
dataset_list = [ 
                 cnstDefs.LETTER_IvsJ,...
                 cnstDefs.LETTER_MvsN,...
                 cnstDefs.LETTER_UvsV,...
                 cnstDefs.LETTER_PvsD,...
                 cnstDefs.LETTER_EvsD, ...
                 cnstDefs.LETTER_EvsF,...
               ];
end
function [dataset_list] = get_SSLds()
global cnstDefs
dataset_list = [ 
                 cnstDefs.SSLBOOK_G241C,...
                 cnstDefs.SSLBOOK_G241N,...
                 cnstDefs.SSLBOOK_COIL,...
                 cnstDefs.SSLBOOK_TEXT,...
                 cnstDefs.SSLBOOK_DIGIT1,...
                 cnstDefs.SSLBOOK_USPS,...
                 cnstDefs.SSLBOOK_COIL2,...
                 cnstDefs.SSLBOOK_BCI,...
               ];
end
function [dataset_list] = get_all_synthesisdatasets_in_alist()
global cnstDefs
dataset_list = [ cnstDefs.SYNTH4OUTLIERORDERED,...%                = 30;
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
cnstDefs.SYNTH4OUTLIER_CORRECTLYLABELED];
end
function [dataset_list] = get_all_DISTsynthesisdatasets_in_alist()
global cnstDefs
dataset_list = [cnstDefs.DIST_SYNTH_MUANDETTOY2, cnstDefs.DIST_SYNTH_MUANDETTOY3, cnstDefs.DIST_SYNTH_MUANDETTOY2, cnstDefs.DIST_SYNTH_MUANDETTOY1, cnstDefs.DIST_SYNTH_MUGAMMA];
end
function [dataset_list] = get_all_DISTdatasets_in_alist()
global cnstDefs
dataset_list   = cnstDefs.CRISIS_ALBERTFLD;
%dataset_list  = [407:419];%cnstDefs.REUTERSIDS];
%dataset_list   = [cnstDefs.IMDBPOLARITY, cnstDefs.AMAZONPOLARITY, cnstDefs.YELPSENT ];
%dataset_list  = cnstDefs.YELPPOLARITY;
%dataset_list  = [cnstDefs.NEWSGROUPIDS];
%dataset_list  = [cnstDefs.NEWSGROUPONEVSRESTIDS];
%dataset_list = [ cnstDefs.NEWSGROUPRANGE, cnstDefs.NEWSGROUP0102, cnstDefs.NEWSGROUP0115, cnstDefs.NEWSGROUP0104, cnstDefs.NEWSGROUP0103, cnstDefs.NEWSGROUP0102, cnstDefs.MUSK1, cnstDefs.MUSK2, cnstDefs.USPSDIST_3_4, cnstDefs.USPSDIST_1_8, cnstDefs.USPSDIST_3_8, cnstDefs.USPSDIST_6_9 ];
end
