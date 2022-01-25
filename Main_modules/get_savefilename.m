function [filename]= get_savefilename(insertdate, path, datasetName, stage, method)
global cnstDefs
if insertdate
   filename = [cnstDefs.main_path,path,'DS=',datasetName,'_STAGE=', stage,'_METHOD=',method,'_', datestr(datetime(),'yyyy_mmmm_dd_HH_MM')];
else
   filename = [cnstDefs.main_path,path,'DS=',datasetName,'_STAGE=', stage,'_METHOD=',method];
end
end