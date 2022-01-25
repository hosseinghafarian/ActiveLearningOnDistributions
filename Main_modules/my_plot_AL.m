function my_plot_AL(almethods_list, figfilename, store_ind_union, ACC_PLOT_Data, method_store_inds, name_store_inds, ind, datasetName)
method_plot_prop = { 
                     {1, 'none',  '-',  'b'},...  %1
                     {1, 'none', '--',  'm'},...  %2
                     {1, 'none',  ':',  'c'},...  %3
                     {1, 'none', '-.',  'r'},...  %4
                     {1, 'none', '--',  'g'},...  %5
                     {1, 'none',  ':',  'b'},...   %6
                     {1, 'none', '-.',  'b'},...   %7
                     {1, 'none', '--',  'k'},...   %8
                     {1, 'none', '-.',  'k'},...  %9
                     {2, 'none',  '-',  'k'},...  %10 
                     {1, 'none', '-.',  'c'},...  %11
                     {1, 'none', '-',   'r'},...  %12
                     {1, 'none', '-',   'k'},...  %13
                     {1, 'none', '-.',  'b'},...   %14
                     {1, 'none', '--',  'w'},...   %15
                     {1, 'none', ':',   'k'},...   %16
                  };
n = numel(almethods_list);
figure;
firstplot = true;
n_i = size(method_store_inds,1);
for i=store_ind_union' 
   found = false;
   for j = 1:n_i
       if ismember(i,method_store_inds(j,:)),
           method_name = almethods_list{j}{6};
           found = true;
           break;
       end
   end
   if ~found, continue; end
   p = plot(ACC_PLOT_Data(i, 1:ind),'DisplayName',method_name);
   p(1).LineWidth = method_plot_prop{i}{1};
   p(1).Marker    = method_plot_prop{i}{2};
   p(1).LineStyle = method_plot_prop{i}{3};
   p(1).Color     = method_plot_prop{i}{4};
   if firstplot
       str  = strcat('dataset:',datasetName);
       title(str);
       xlabel('Number of queries');
       ylabel('Accuracy');
       hold on;
       firstplot = false;
   end
end
% ylim_a = ylim;
xlim([1 ind]);
pbaspect([1.0000    0.8989    0.8989]);
figfilename = strcat(figfilename, '.fig');
savefig(figfilename);
% export_fig('test.png');
% ylim(ylim_a);
end