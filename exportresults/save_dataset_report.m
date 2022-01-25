function save_dataset_report(csvfilefulpath, myprofile)
global cnstData
global cnstDefs

global cnstQTIMECSVfile
global crisisDInd

fid = fopen(csvfilefulpath, 'wt');
fprintf(fid, 'Dataset Name, Size, Total Instances, Average Inst, Min Inst, Max Inst, Std Inst, Labeled Inst Per, Std Num of Labeled Inst, Min Num Lab Inst, Max Num Lab Inst \n');
for datasetId = myprofile.dataset_list
    for crisisInd = 1:24
        try 
            crisisDInd = crisisInd;
            [data ]    = load_dataobject(datasetId, myprofile);
            [ds_report] = data.summarizedinfo(data);
            fprintf(fid, '%s, %d, %d, %4.2f, %d, %d, %4.2f, %4.2f, %4.2f, %d, %d\n',... 
                             ds_report.name, ds_report.n, ds_report.dist.totalninst, ds_report.dist.averageninst,...
                             ds_report.dist.min_ninst, ds_report.dist.max_ninst, ds_report.dist.stdninst, ...
                             ds_report.dist.n_l2_labper, ds_report.dist.n_lab_perdist_std, ...
                             ds_report.dist.n_lab_perdist_min, ds_report.dist.n_lab_perdist_max);
        catch
           disp('oh'); 
        end    
    end
end
fclose(fid);
end