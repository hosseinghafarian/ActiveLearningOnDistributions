function my_export_plot_AL(exp_par_name, figfilename, ACC_PLOT_Data, datasetName, xlabelstr, ylabelstr, usedatasetName)
if nargin<7
    usedatasetName = false;
end
global cnstDefs
if nargin <= 6
    xlabelstr = 'Number of queries';
    ylabelstr = 'Accuracy';
end
firstplot = true;

ind           = size(ACC_PLOT_Data, 2);
numPlotPoints = floor(ind);
color_array = { 
               [ 0 0 0],...
               [ 0 0 1],...
               [ 0 0.6 0],...
               [ 1 0 0],...
               [ 1 0 1],...
               [ 0 1 1],...
               [ 0.6 0.6 0],...
               };
BLACK  = 1; BLUE   = 2; GREEN  = 3; RED    = 4; MAGENTA= 5; CYAN   = 6; YELLOW = 7;
marker_array = {'none','o','+','*','.','x','s','d','^','v','>','<','p','h'};
NONE   = 1; CIRCLE = 2; PLUS   = 3; STAR   = 4; POINT  = 5; CROSS  = 6; SQUARE = 7; DIAMOND= 8;
UPTRIAG=9;  DOTRIAG = 10; RITRIAG = 11; LETRIAG = 12; PENTAG  = 13; HEGAX   = 14; 
linestyle_array = {'-','--','-.',':'};
SOLID  = 1; DASHED = 2; DOTTED = 3; DASHDOT= 4;

method_plot_prop = cell(20,1);
rank = [2,1,4,3,5,6,7,8,9,10,11,12,13,14, 15,16,17,18,19,20];
method_plot_prop{rank(1)} = {2,  NONE, SOLID  ,  BLACK};
method_plot_prop{rank(2)} = {1,  NONE, DASHED ,  BLUE};
method_plot_prop{rank(3)} = {1,  NONE, DOTTED ,  GREEN};
method_plot_prop{rank(5)} = {1,  NONE, DASHDOT,  RED};
method_plot_prop{rank(4)} = {1,  CIRCLE, SOLID  ,  MAGENTA};
method_plot_prop{rank(6)} = {1,  PLUS, DASHDOT,  CYAN};
method_plot_prop{rank(7)} = {1,  STAR, DOTTED,  BLACK};
method_plot_prop{rank(8)} = {1,  SQUARE, DASHDOT,  BLUE};
method_plot_prop{rank(9)} = {1,  CROSS, DASHED,  GREEN};
method_plot_prop{rank(10)} = {1,  CIRCLE, DASHDOT,  RED};
method_plot_prop{rank(11)} = {1,  PLUS, DOTTED,  MAGENTA};
method_plot_prop{rank(12)} = {1,  STAR, DASHDOT, CYAN};
method_plot_prop{rank(13)} = {1,  SQUARE, DASHDOT,  BLACK};
method_plot_prop{rank(14)} = {1,  CROSS, DOTTED,  BLUE};
method_plot_prop{rank(15)} = {1,  POINT,DOTTED,  BLACK};
method_plot_prop{rank(16)} = {1,  PLUS, SOLID ,  GREEN};
method_plot_prop{rank(17)} = {1,  STAR, SOLID,   RED};
method_plot_prop{rank(18)} = {1,  SQUARE, DASHDOT,  BLACK};
method_plot_prop{rank(19)} = {1,  CROSS, DOTTED,  RED};
% method_plot_prop{rank(20)} = {1,  PENTAG, DOTTED,  MAGENTA};
% method_plot_prop{rank(5)} = {1,  CIRCLE, SOLID  ,  MAGENTA};
% method_plot_prop{rank(6)} = {1,  PLUS, DASHDOT,  CYAN};
% method_plot_prop{rank(7)} = {1,  STAR, DOTTED,  BLACK};
% method_plot_prop{rank(8)} = {1,  SQUARE, DASHDOT,  BLUE};
% method_plot_prop{rank(9)} = {1,  CROSS, DASHED,  GREEN};
% method_plot_prop{rank(10)} = {1,  CIRCLE, DASHDOT,  RED};
% method_plot_prop{rank(11)} = {1,  PLUS, DOTTED,  MAGENTA};
% method_plot_prop{rank(12)} = {1,  STAR, DASHDOT, CYAN};
% method_plot_prop{rank(13)} = {1,  SQUARE, DASHDOT,  BLACK};
% method_plot_prop{rank(14)} = {1,  CROSS, DOTTED,  BLUE};
% method_plot_prop{rank(15)} = {2,  POINT,DOTTED,  BLACK};
% method_plot_prop{rank(16)} = {1,  PLUS, SOLID ,  GREEN};
% method_plot_prop{rank(17)} = {1,  STAR, SOLID,   RED};
% method_plot_prop{rank(18)} = {1,  SQUARE, DASHDOT,  BLACK};
% method_plot_prop{rank(19)} = {1,  CROSS, DOTTED,  RED};
% method_plot_prop{rank(20)} = {1,  PENTAG, DOTTED,  MAGENTA};
hFig = figure;
hAxes = axes('Parent',hFig,...
    'FontSize',12,'XGrid','on', 'YGrid','on', 'Box', 'on',...
    'TickLabelInterpreter', 'latex');
hold('on');

plotInd = floor(linspace(1, ind, numPlotPoints));


n_i = numel(exp_par_name);
for i=1:n_i
   method_name = exp_par_name{i};
   %p = plot(ACC_PLOT_Data(i, 1:ind),'DisplayName',method_name);
   p = plot(hAxes, ACC_PLOT_Data(i, plotInd),'DisplayName',method_name);
   p(1).LineWidth = method_plot_prop{i}{1};
   p(1).Marker    = marker_array{method_plot_prop{i}{2}};
   p(1).LineStyle = linestyle_array{method_plot_prop{i}{3}};
   p(1).Color     = color_array{method_plot_prop{i}{4}};
   %p(1).xlim      = 
   if firstplot
       str  = strcat('dataset:',datasetName);
       title(str);
       xlabel(xlabelstr, 'interpreter', 'Latex');
       ylabel(ylabelstr, 'interpreter', 'Latex');
       hold on;
       firstplot = false;
   end
end
legend('show','Location','southeast');
% ylim_a = ylim;
xlim([1 ind]);
pbaspect([1.0000    0.8989    0.8989]);
figfilename = strcat(figfilename, '.fig');
if usedatasetName
    imagefilename = strcat(datasetName,'.eps');
else
    imagefilename = strcat(figfilename, '.eps');
end
savefig(figfilename);
%export_fig(imagefilename);
saveas(gcf,imagefilename, 'epsc');
% ylim(ylim_a);
end
% This script removes the selected lines from a figure's legend. The lines
% must be selected BEFORE calling this script
%
% Created June 6, 2014, by Adam Noel
%
function removeLegendLines()
a = findobj('Selected', 'on');
for i = 1:length(a)
    hasbehavior(a(i), 'legend', false);
end
end