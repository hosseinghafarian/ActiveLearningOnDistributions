doload = true;
if doload
    load('2dsimulations\2ddata.mat', 'outp');
else
    [ fig, outp, X] = sampleShow('2D data');
    save('2ddata.mat', 'outp');
end
X = outp(1:2,:);
label = outp(3,:);
dist = pdist2(X', X');
ND  = size(dist ,1);
percent  =2;
[rho, delta, ordrho, nneigh, dc, efrho] = fastcluster_deltarho2(dist, percent);
deltamax  = max(delta);
rhomax    = max(rho);
deltamin  = 0.4*deltamax;
rhomin    = 0.4*rhomax;
[NCLUST, cl, icl, halo] = fastcluster_assign(ND, dist, dc, rho, delta, ordrho, nneigh, deltamin, rhomin);

disp('Generated file:DECISION GRAPH')
disp('column 1:Density')
disp('column 2:Delta')

fid = fopen('DECISION_GRAPH', 'w');
for i=1:ND
   fprintf(fid, '%6.2f %6.2f\n', rho(i),delta(i));
end
fprintf('NUMBER OF CLUSTERS: %i \n', NCLUST);
for i=1:NCLUST
  nc=0;
  nh=0;
  for j=1:ND
    if (cl(j)==i) 
      nc=nc+1;
    end
    if (halo(j)==i) 
      nh=nh+1;
    end
  end
  fprintf('CLUSTER: %i CENTER: %i ELEMENTS: %i CORE: %i HALO: %i \n', i,icl(i),nc,nh,nc-nh);
end

disp('Performing assignation')
plot2d_clusters(ND, NCLUST, dist, halo, icl, delta, rho);
%for i=1:ND
%   if (halo(i)>0)
%      ic=int8((halo(i)*64.)/(NCLUST*1.));
%      hold on
%      plot(Y1(i,1),Y1(i,2),'o','MarkerSize',2,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
%   end
%end
faa = fopen('CLUSTER_ASSIGNATION', 'w');
disp('Generated file:CLUSTER_ASSIGNATION')
disp('column 1:element id')
disp('column 2:cluster assignation without halo control')
disp('column 3:cluster assignation with halo control')
for i=1:ND
   fprintf(faa, '%i %i %i\n',i,cl(i),halo(i));
end