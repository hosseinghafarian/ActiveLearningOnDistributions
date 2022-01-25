function testnew

%  Robust Classifier :(mytwo func classifier). SVM Classification 2D examples

%% Initialization 
opmode=1;% 1: ordinary , 2: Active learning 

%-------------Data Set-------------------------------------

showOptions.clear             = true;
showOptions.acceptNewInstance = true;
showOptions.selectInstance    = false;
showOptions.StopforKey        = false;
showOptions.isUpdate          = true;
showOptions.onSpecificFigure  = false;
showOptions.showContour       = true;
showOptions.showIndex         = false;
%% data file
%% Results: 
% #1: Good,type 5:AlternateOptimization,datafile='orderedlittleOutlier2t2.mat'; 
%     lambda = 4, lambda_o = 0.1, dSigma=0.2*refmed
% #2: Good,type 5:AlternateOptimization,datafile='orderedlittleOutlier2t2.mat'; 
%     lambda = 2.5, lambda_o = 0.1, dSigma=0.2*refmed
% #3: Good,type 7:ConvexRelax,datafile='orderedlittleOutlier2t2.mat'; 
%     lambda = 2.5, lambda_o = 0.1, dSigma=0.2*refmed
% #4: Good,type 7:ConvexRelax,datafile='orderedlittleOutlier2t2.mat'; 
%     lambda = 7, lambda_o = 0.1, dSigma=0.2*refmed
% #5: Good,type 7:ConvexRelax,datafile='orderedlittleOutlier2t2.mat'; 
%     lambda = 7, lambda_o = 0.1, dSigma=0.2*refmed
% #6: NotSoGood,type 7:ConvexRelax,datafile='orderedlittleOutlier2t2.mat'; 
%     lambda = 15, lambda_o = 0.1, dSigma=0.2*refmed, Classifier is too
%     simple for w
% #7: Good,type 7:ConvexRelax,datafile='orderedlittleOutlier2t2.mat'; 
%     lambda = 1, lambda_o = 0.1, dSigma=0.2*refmed
% #8: Bad for Outlier,type 5,datafile='orderedlittleOutlier2t2.mat'; 
%     lambda = 1, lambda_o = 0.1, dSigma=0.2*refmed
%     This is exactly the same as #7, but with alternating optimization, 
%     it selects the most closest pair of points with opposite class (near the border) 
% #9: Good,type 5:AlternateOptimization,datafile='outlier4orderedlittle.mat'; 
%     lambda = 2.5, lambda_o = 0.1, dSigma=0.2*refmed
% #10:Good,type 7:ConvexReleax ,datafile='outlier4orderedlittle.mat'; 
%     lambda = 2.5, lambda_o = 0.1, dSigma=0.2*refmed
% #11:Ok,Mixedresult,type 5: ,datafile='outlier4orderedlittle.mat'; 
%     lambda = 2.5, lambda_o = 0.1, dSigma=0.2*refmed
% #12:Good,type 7: ,datafile='outlier4orderedlittle.mat'; 
%     lambda = 2.5, lambda_o = 0.1, dSigma=0.2*refmed
%     It is better than #11, the alternating optimization counterpart in
%     terms of selecting noisypoints but it is as always less sharp in
%     terms of values of w_o at those points and so, the border is more affected
%     by noisy points in simple classifier than in alternating optimization
% #13:Good,type 7: ,datafile='outlier4orderedlittle.mat'; 
%     lambda = 2.5, lambda_o = 0.1, dSigma=0.2*refmed, cp=0.5, cp is:
%     (objective+cp*sum(p) instead of sum(p)<= n_o as a constraint)
%     This method works better #12, it't border is not affected by either
%     outliers or label noisy points. 
% #14:Good,type 7: ,datafile='outlier4orderedlittle.mat'; 
%     lambda = 2.5, lambda_o = 0.1, dSigma=0.2*refmed, cp=0.8, cp is:
%     (objective+cp*sum(p) instead of sum(p)<= n_o as a constraint)
%     This method works better than#12. it't border is affected a little more by 
%     outliers and/or label noisy points. it seems that it may be a better result
%     compared to other type 7 experiment #13,#15
% #15:MixedResult,type 7: ,datafile='outlier4orderedlittle.mat'; 
%     lambda = 2.5, lambda_o = 0.1, dSigma=0.2*refmed, cp=0.8, cp is:
%     (objective+cp*sum(p) instead of sum(p)<= n_o as a constraint)
%     This method works better than#12. it't border is affected a little more by 
%     outliers and/or label noisy points. Compared to #13 In terms of 
%     p_i values of none noisy points are closer to zero than #13. But border is 
%     affected by label noisy points. ( point upright in the fig 14,
%     painted in blue
%% 
reWriteData=0;
% datafile='testSimplef3.mat';
% datafile='orderedlittle.mat';
%datafile='outlier4orderedlittle.mat';
%datafile='littlef.mat'
%datafile='outlier4orderedlittleplusmanydata.mat';
%datafileout='orderedlittleOutlier2t1.mat';
%datafile='order4outlierLambdaExp.mat'
%datafile='orderedlittleOutlier2t2.mat';
%datafile='testSimplef.mat';
%datafile='orderedOutlier2t2.mat';
%Experiments for the paper:
for expnum = 1:10 
    close all
    switch expnum
    case  1
    %-------------------Experiment For the Paper------------------------------
    dname    = 'testSimplef2';
    n_o      = 4;
    lambda   = 2.5;
    lambda_o = 0.025;
    timesmedheur = 1;
    case 2 
    dname    = 'testSimplef';
    n_o      = 4;
    lambda   = 2.5;
    lambda_o = 0.025;
    timesmedheur = 1;
    case 3
    dname    = 'testSCCOutsidesame';
    n_o      = 4;
    lambda   = 2.5;
    lambda_o = 0.025;
    timesmedheur = 1;
    case 4
    dname    = 'testSimplef3';
    n_o      = 4;
    lambda   = 2.5;
    lambda_o = 0.025;
    timesmedheur = 1; 
    case 5
    dname    = 'testtwofuncCCSC';
    n_o      = 4;
    lambda   = 2.5;
    lambda_o = 0.025;
    timesmedheur = 1; 
    case 6
    dname    = 'testtwofuncCCSC';
    n_o      = 2;
    lambda   = 0.1;
    lambda_o = 0.0025;
    timesmedheur = 1; 
    case 7
    dname    = 'testtwofuncOutSideInv';
    n_o      = 2;
    lambda   = 0.1;
    lambda_o = 0.0025;
    timesmedheur = 1;
    case 8
    dname    = 'testtwofuncOutsideSame';
    n_o      = 2;
    lambda   = 0.1;
    lambda_o = 0.0025;
    timesmedheur = 1;
    case 9
    dname    = 'testtwofuncOutsideSame2point';
    n_o      = 2;
    lambda   = 0.1;
    lambda_o = 0.0025;
    timesmedheur = 1;
    case 10
    dname    = 'testtwofuncOutsideSame4point';
    n_o      = 4;
    lambda   = 0.1;
    lambda_o = 0.0025;
    timesmedheur = 1;

    end
    stname   = sprintf('no=%d,lam=%5.3f,lamo=%5.4f',n_o,lambda, lambda_o);
    stname = strrep(stname, '.', 'dot');
    figsave  = [dname, stname];
    datafile = [dname, '.mat'];

    %datafile = 'testtwofuncOutsideSame4point.mat';
    %     
    drawclassifiercountour(datafile, figsave, showOptions, n_o, lambda, lambda_o, timesmedheur);
end
end
function drawclassifiercountour(datafile, figsave, showOptions, n_o, lambda, lambda_o, timesmedheur)
wlambda = 0.1;
figName = 'samples';
reWriteData = 0;
hf(1) = figure;
if ~exist(datafile, 'file') || reWriteData
    XClass = sampleShow(figName);
    XClass = XClass';
    save(datafile,'XClass');    
else
   load(datafile);
   nc=size(XClass,1);
   sampleShow_old(figName,XClass(:,1:2)',XClass(:,3)',[1:nc],showOptions);
%     XClass = XClass';
%     save(datafileout,'XClass');
end
fig = gcf;
fig.InvertHardcopy = 'off';
saveas(gcf, [figsave,'_data'], 'epsc');
%% 
initL = [1,2];%initperm(1:startupsize);
xapp = XClass(:,1:2);
yapp = XClass(:,3);
%[xapp,yapp]=datasets('Checkers',nbapp,nbtest,sigma);
%-----------------------------------------------------
%   Learning and Learning Parameters
rho       = 10;
type1      = 7;
type2      = 2;

distpoints = pdist2(xapp,xapp);
refmed     = median(median(distpoints));
dSigma     = timesmedheur*refmed;
Ke         = exp(- pdist2(xapp,xapp)/(2*dSigma^2)) ;

n_all      = size(xapp,1);
n_s        = n_all - n_o;

%----------------------Calling Classifiers----------------------
brdr=1;
[BoxXY]=findAxis(xapp,brdr);
[retResult,info1,info2]=clresult(Ke,type1,type2,xapp,yapp,lambda,lambda_o,wlambda,BoxXY,initL,Ke,...
                                 dSigma,n_o);
%----------------------Saving data------------------------------

hf(2) = figure;
colormap(jet);
ttile='Simple classifier';
info1.normW=0;
info1.normW_o=0;
dataContour(retResult.xtesta1,retResult.xtesta2,retResult.xtest, ...
            retResult.ypredmat,retResult.ypred,...
            lambda,info1.normW,ttile,...
            xapp,yapp);
fig = gcf;
fig.InvertHardcopy = 'off';        
saveas(gcf, [figsave,'_simp'], 'epsc');
hf(3) = figure;
ttile='Complex classifier';
dataContour(retResult.xtesta1,retResult.xtesta2,retResult.xtest, ...
            retResult.ypredmat_o,retResult.ypred_o,...
            lambda_o,info1.normW_o,ttile,...
            xapp,yapp);
saveas(gcf, [figsave,'_cmplx'], 'epsc');
ttile='Ordinary classifier';
info2.normW=0;
hf(4) = figure;
dataContour(retResult.xtesta1,retResult.xtesta2,retResult.xtest, ...
            retResult.ypredmatsf,retResult.ypredsf,...
            lambda,info2.normW,ttile,...
            xapp,yapp);
fig = gcf;
fig.InvertHardcopy = 'off';        
saveas(gcf, [figsave,'_ord'], 'epsc');        
showTrPointResult(info1);
    
savefig(hf, [figsave,'.fig']);
    
function [retResult,info1,info2]=clresult(K,type1,type2,xapp,yapp,lambda,lambda_o,wlambda,BoxXY,initL,K_o,Sigma,n_o)
    epsilon = .000001;
    kerneloption= 1;
    kernel='gaussian';
    
    [info1]=twofuncsvm(K,type1,xapp,yapp,lambda,lambda_o,wlambda,epsilon,kernel,kerneloption,rho,initL,K_o,n_o);

    [info2]=twofuncsvm(K,type2,xapp,yapp,lambda,lambda_o,wlambda,epsilon,kernel,kerneloption,rho,initL,K_o,n_o);

    %info2=info1;
    %% --------------Testing Generalization performance ---------------
    %retResult=1;
    reso= 0.05;
    [xtesta1,xtesta2]=meshgrid([BoxXY.xmina:reso:BoxXY.xmaxa],[BoxXY.ymina:reso:BoxXY.ymaxa]);
    [na,nb] =size(xtesta1);
    xtest1=reshape(xtesta1,1,na*nb);
    xtest2=reshape(xtesta2,1,na*nb);
    xtest=[xtest1;xtest2]';
    [ret_ypred] = mytwofuncsvmval(Sigma,xtest,xapp,yapp,info1,lambda,lambda_o,wlambda);
    
    %[lossVec] = comploss(K,xapp,yapp,info1,info2,lambda,lambda_o);
    ypredmat  =reshape(ret_ypred.ypred,na,nb);
    ypredmat_o=reshape(ret_ypred.ypred_o,na,nb);
    if ret_ypred.type==4
        ypredmatf = reshape(ret_ypred.ypredf,na,na);
    else 
        ypredmatf = ypredmat;
    end
    retResult.xtest    = xtest;
    retResult.xtesta1  = xtesta1;
    retResult.xtesta2  = xtesta2;

    retResult.ypredmat  =ypredmat;
    retResult.ypredmat_o=ypredmat_o;

    retResult.ypredmatf =ypredmatf;

    retResult.ypred     = ret_ypred.ypred;
    retResult.ypred_o   = ret_ypred.ypred_o;
    
    [ret_ypred2] = mytwofuncsvmval(Sigma,xtest,xapp,yapp,info2,lambda,lambda_o,wlambda);
    %ret_ypred2=ret_ypred;
    ypredmatsf = reshape(ret_ypred2.ypred,na,nb);
    retResult.ypredmatsf= ypredmatsf;
    retResult.ypredsf   = ret_ypred2.ypred;
    %retResult.lossVec   = lossVec;
end
function dataContour(xtesta1,xtesta2, xtest, ypredmat,ypred,lambda,normW,ttile,xapp,yapp)
        ind1 =find(ypred>0);
        indm1=find(ypred<0);
        cla;
        hold on ;
        %colormap hsv;
        colormap jet;
        contourf(xtesta1,xtesta2,ypredmat,20); 
%         xte1 = xtest(:,1);
%         xte2 = xtest(:,2);
%         set(gca,'ydir','normal');
        %set(gca,'xdir','reverse')
%         imagesc(xte1, xte2, ypred) ; 
        shading interp
%         shading flat;
        hold on
        %[cs,h]=contour(xtesta1,xtesta2,ypredmat,[-1 0 1],'k');
        [cs,h]=contour(xtesta1,xtesta2,ypredmat,[0 0]);
        set(h,'color', 'b', 'linewidth', 4) ;
        shading interp
        %%
    %     clabel(cs,h);
    %     h1=plot(xapp(yapp==1,1),xapp(yapp==1,2),'+k'); 
    %     set(h1,'LineWidth',2);
    % 
    %     h2=plot(xapp(yapp==-1,1),xapp(yapp==-1,2),'xk'); 
    %     set(h2,'LineWidth',2);
    %     h3=plot(xapp(:,1),xapp(:,2),'ok'); 
    %     set(h3,'LineWidth',2);
%%
%         showpoints(xapp,yapp);
        shading interp
        lambtex=sprintf('\\lambda=%0.5f',lambda);
        %normtex =sprintf('norm W:%0.5f',normW);
        title(ttile, 'Interpreter','latex');
        %ylabel(normtex);
        xlabel(lambtex, 'Interpreter','tex');
end
function dataSurf(xtesta1,xtesta2,ypredmat,ypredmat_o,ypredmatsf)
    hf=gcf;

    figure('Name','Simple Classifier');
    surf(xtesta1,xtesta2,ypredmat);
    hold on;
    figure('Name','Complex Classifier');
    surf(xtesta1,xtesta2,ypredmat_o);

    figure('Name','Ordinary Classifier');
    surf(xtesta1,xtesta2,ypredmatsf);
    figure(hf);
end
function [lossVec] = comploss(K,xapp,yapp,info1,info2,lambda,lambda_o)
    n =size(xapp,1);
    alpha_t=info1.alpha_r;
    alpha_sf = info2.alpha_r;
    hr=info1.h;
    rr=info1.r;
    lossVec.fxi = 1/(2*lambda)*sum((alpha_t.*yapp)'*K,1)';
    lossVec.foxi= 1/(2*lambda_o)*sum((alpha_t.*yapp+hr-rr)'*K,1)';
    lossVec.fxisf = 1/(2*lambda)*sum((alpha_sf.*yapp)'*K,1)';
    lf = 1-yapp.*(lossVec.fxi+lossVec.foxi);
    lossVec.lossf = max(lf,zeros(n,1));
    lossVec.lossfsf = max(1-yapp.*lossVec.fxisf,zeros(n,1));
end
function showlossVec(lossVec)
    n=size(lossVec.fxi,1);
    subplot(5,1,1);
    ylabel('func f');
    plot([1:n],lossVec.fxi);
    subplot(5,1,2);
    ylabel('func fo');
    plot([1:n],lossVec.foxi);
    subplot(5,1,3);
    ylabel('func single f');
    plot([1:n],lossVec.fxisf);
    subplot(5,1,4);
    ylabel('loss f+fo');
    plot([1:n],lossVec.lossf);
    subplot(5,1,5);
    ylabel('loss single f');
    plot([1:n],lossVec.lossfsf);        
end
end
function showTrPointResult(info)
    h=gcf;
    info.wTx_i=info.primalwxT_i;
    n=size(info.wTx_i,2);
    if info.type~=4 
       info.fTx_i=info.wTx_i; % it is set to make it work for types other than 4, for 4 this line must be commented.
    end
    if info.primal==true
        info.wTx_i=info.primalwxT_i;
        info.w_oTx_i=info.primalw_oxT_i;
    end
    figure;
    plot(1:n,info.wTx_i','r',1:n,info.w_oTx_i','b--o',1:n,info.fTx_i','c*');

    
    figure(h);
end
function [BoxXY]=findAxis(X,brdr);
X=X';
BoxXY.xmina=floor(min(X(1,:)))-brdr; %-4;
BoxXY.xmaxa=ceil(max(X(1,:)))+brdr;  % 4 
BoxXY.ymina=floor(min(X(2,:)))-brdr; %-4 
BoxXY.ymaxa=ceil(max(X(2,:)))+brdr;  % 4 
end
function showcontour(boxXY,showOptions,X,y, func_contour, model, learningparams)

ur = linspace(boxXY.xmina,boxXY.xmaxa,256) ;
vr = linspace(boxXY.ymina,boxXY.ymaxa,256) ;
gca
[u,v] = meshgrid(ur,vr) ;
X_dense = [v(:)' ; u(:)'] ;
f_dense = feval(func_contour, model, learningparams, X_dense);
f_dense = reshape(f_dense, size(u,2), size(u,1)) ;
cla ;
%set(gca,'xdir','reverse')

imagesc(vr,ur,f_dense) ; 
set(gca,'ydir','normal');
colormap hsv;%cool ; 
hold on ;
% [c,hm] = contour(ur,ur,f_dense,[-1 -1]) ;
% set(hm,'color', 'r', 'linestyle', '--') ;
% [c,hp] = contour(ur,ur,f_dense,[+1 +1]) ;
% set(hp,'color', 'g', 'linestyle', '--') ;
[c,hz] = contour(ur,ur,f_dense,[0 0]) ;
set(hz,'color', 'b', 'linewidth', 4) ;
hg  = plot(X(1,y>0), X(2,y>0), 'g.', 'markersize', 10) ;
hr  = plot(X(1,y<0), X(2,y<0), 'r.', 'markersize', 10) ;
% hko = plot(X(1,model.svind), X(2,model.svind), 'ko', 'markersize', 5) ;
% hkx = plot(X(1,model.bndind), X(2,model.bndind), 'kx', 'markersize', 5) ;
end
function showpoints(X,class,showIndex,t)
    typ = 2*class;
    x=X(:,1);
    y=X(:,2);
%     TL=type(initL);
    
    if nargin <= 3
        showIndex = false;
    else if nargin >4
             inTex = num2str(t);
             
        end
    end
    if size(x,1)==1 && size(y,1)==1  
        showPointColor(typ,x,y);
        if showIndex 
            text(x,y,inTex,'fontsize',7)
        end
        return
    end
    for i=1:size(x,1)
        showPointColor(typ(i),x(i),y(i))
        inTex = int2str(i);
        if showIndex
            text(x(i),y(i),inTex,'fontsize',7);
        end
    end

end
function showPointColor(typ,x,y)
    switch (typ)
            case -2
                plot(x,y,'r*', 'markersize', 20);
            case -1  % Class 2 (-)
                plot(x,y,'r.', 'markersize', 10);
                
            case 0   % Unlabeled data
                plot(x,y,'b.','markersize',10);
            case 1   % Class 1 (+)
                plot(x,y, 'g.', 'markersize', 10);
            case 2   
                plot(x,y,'g+','markersize',20);
            case 3 
                plot(x,y,'* black');
            otherwise 
                disp('Error data type');
        end
end
