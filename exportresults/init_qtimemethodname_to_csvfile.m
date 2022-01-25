function init_qtimemethodname_to_csvfile(firstcolumn, methods_list)
global cnstCSVfilePath
global cnstQTIMECSVfile
fid = fopen([cnstCSVfilePath,'\',cnstQTIMECSVfile,'.csv'], 'wt');
nc = numel(methods_list);
fprintf(fid,'%s,', firstcolumn);
for i=1:nc-1
   fprintf(fid,'%s,', [methods_list{i}]);
end
fprintf(fid,'%s\n', [methods_list{nc}]);
fclose(fid);
end