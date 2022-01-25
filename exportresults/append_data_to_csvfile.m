function append_data_to_csvfile(trainsize, measures)
global cnstCSVfilePath
global cnstCSVfile
fid = fopen([cnstCSVfilePath,'\',cnstCSVfile], 'at');
nrow = numel(measures.acc_avg);
fprintf(fid, '%5d,', trainsize);
for j=1:nrow-1
   fprintf(fid, '%8.5f,%8.5f,', measures.acc_avg(j), measures.acc_std(j));
end
fprintf(fid, '%8.5f,%8.5f\n', measures.acc_avg(nrow), measures.acc_std(nrow));
fclose(fid);
end