function ACC_PLOT_Data                 = computeAverage(accuracy,exp_num)
Acc = sum(accuracy,2)/exp_num;
ACC_PLOT_Data = permute(Acc, [ 3 1 2]);
Stdcc = std(accuracy, 0, 2) 
STDCC_PLOT_DATA = permute(Stdcc, [3 1 2]);
end
%     ACC_PLOT_Data = zeros(method_num,query_num);
%     for i=1:method_num
%         ACC_PLOT_Data(i,:)= 0;
%         varexp_num = zeros(1,query_num);
%         for j=1:exp_num
%             acc = reshape(accuracy(:,j,i),1,query_num);
%             %varexp_num  = varexp_num + acc~=0;
%             ACC_PLOT_Data(i,:) = ACC_PLOT_Data(i,:) + acc;
%         end
%         ACC_PLOT_Data(i,:)= ACC_PLOT_Data(i,:) / exp_num;
%     end
