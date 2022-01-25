function init_methodname_to_csvfile(firstcolumn, Classification_exp_param)
global cnstCSVfilePath
global cnstCSVfile
fid = fopen([cnstCSVfilePath,'\',cnstCSVfile], 'wt');
nc = numel(Classification_exp_param);
fprintf(fid,'%s,', firstcolumn);
for i=1:nc-1
   fprintf(fid,'%s,%s,', [Classification_exp_param{i}.clmethod, '_avg'], [Classification_exp_param{i}.clmethod, '_std']);
end
fprintf(fid,'%s,%s\n', [Classification_exp_param{nc}.clmethod, '_avg'], [Classification_exp_param{nc}.clmethod, '_std']);
fclose(fid);
end