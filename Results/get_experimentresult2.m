function [T] = get_experimentresult(resmatfile)
load(resmatfile);
dn = strsplit(data.datasetName, '='); 
datasetname = dn{1};
methodname = measures.clmethods;
n_method = numel(methodname);


submethod{1,1} = 'KNNfidf';
submethod{1,2} = 'KNN';
submethod{2,1} = 'DecisionTreetfidf';
submethod{2,2} = 'DecisionTree';
submethod{3,1} = 'NaiveBayestfidf';
submethod{3,2} = 'NaiveBayes';
submethod{4,1} = 'SVMtfidf';
submethod{4,2} = 'SVM';
n_subm         = 4;
for i = 1:n_method
   for j = 1:n_subm
       if strcmp(methodname{i}, submethod{j,1})
           methodname{i} = submethod{j,2};
           break;
       end
   end
end

colname = {'acc_avg', 'acc_std', ...
           'prec_avg', 'prec_std',...
           'recal_avg', 'recal_std',...
           'spec_avg', 'spec_std', ...
           'f1score_avg', 'f1score_std'};
columnname = {'Accuracy', 'AccuracySTD', ...
           'Precision', 'PrecisionStd',...
           'Recal', 'RecalStd',...
           'Specificity', 'SpecificityStd', ...
           'FOnescore', 'FOnescoreStd'};

n_col    = numel(colname);
for j = 1:n_col
   maxrow = 1;
   for i = 2:n_method
       if (measures.experiments_details{i,1}.(colname{j}) > measures.experiments_details{maxrow,1}.(colname{j}))
           maxrow = i;
       end
   end
   maxrowcol(j) = maxrow;
end

for i = 1:n_method
   for j = 1:n_col
       if rem(j,2)==1
          datamatrix{i,j} = chechNANandbestval(measures.experiments_details{i,1}.(colname{j}), (i==maxrow(j)));
       else
          datamatrix{i,j} = sprintf('%4.1f', measures.experiments_details{i,1}.(colname{j}));
       end
   end
end
% T = table(datamatrix(:,1), datamatrix(:,2), datamatrix(:,3), datamatrix(:,4), datamatrix(:,5),...
%           datamatrix(:,6), datamatrix(:,7), datamatrix(:,8), datamatrix(:,9), datamatrix(:,10), 'VariableNames', columnname, 'RowNames', methodname);
T = cell2table(datamatrix, 'VariableNames', columnname, 'RowNames', methodname);
      
writetable(T,[datasetname, '_result.csv'],'Delimiter',',', 'WriteRowNames',true);
end
function [valstr] = chechNANandbestval(value, bestval)
    if isnan(value)
         valstr = sprintf('-');
    elseif bestval
       valstr = sprintf('\\textbf{%5.2f}', value);
    else
        valstr = sprintf('%5.2f', value);
    end    
end