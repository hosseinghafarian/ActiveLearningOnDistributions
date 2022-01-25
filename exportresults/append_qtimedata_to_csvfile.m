function append_qtimedata_to_csvfile(datasetname, timeavgmethod)
global cnstCSVfilePath
global cnstQTIMECSVfile
fid = fopen([cnstCSVfilePath,'\',cnstQTIMECSVfile,'.csv'], 'at');
nrow = numel(timeavgmethod);
fprintf(fid, '%s,', datasetname);
for j=1:nrow-1
   fprintf(fid, '%8.5f,', timeavgmethod(j));
end
fprintf(fid, '%8.5f\n', timeavgmethod(j+1));
fclose(fid);
end