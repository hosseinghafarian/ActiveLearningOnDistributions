function exclass

%  Robust Classifier :(mytwo func classifier). SVM Classification 2D examples

%% Initialization 
opmode=1;% 1: ordinary , 2: Active learning 
close all
%-------------Data Set-------------------------------------
nbapp=200;
nbtest=0;
sigma=2;
figName = 'samples';
verbose=0;
showOptions.clear=true;
showOptions.acceptNewInstance=true;
showOptions.selectInstance = false;
showOptions.StopforKey     = true;
showOptions.isUpdate = true;
showOptions.onSpecificFigure=false;
showOptions.showContour = false;
showOptions.showIndex   = false;
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
reWriteData=0;
%datafile='testSimplef3.mat';
%datafile='orderedlittle.mat';
datafile='outlier4orderedlittle.mat';
%datafile='littlef.mat'
%datafile='outlier4orderedlittleplusmanydata.mat';
%datafileout='orderedlittleOutlier2t1.mat';
datafile='order4outlierLambdaExp.mat'
%datafile='orderedlittleOutlier2t2.mat';
%datafile='testSimplef.mat';
%datafile='orderedOutlier2t2.mat';

if ~exist(datafile, 'file') || reWriteData
    XClass = sampleShow(figName);
    XClass = XClass';
    save(datafile,'XClass');    
else
    load(datafile);
   nc=size(XClass,1);
sampleShow(figName,XClass(:,1:2)',XClass(:,3)',[1:nc],showOptions);
%     XClass = XClass';
%     save(datafileout,'XClass');
end
%% 
nc=size(XClass,1);
initperm=randperm(nc);
startupsize=2;

initL = [1,2];%initperm(1:startupsize);

xapp = XClass(:,1:2);
yapp = XClass(:,3);
%[xapp,yapp]=datasets('Checkers',nbapp,nbtest,sigma);

%-----------------------------------------------------
%   Learning and Learning Parameters
hn=3;
vn=5;

i=1;j=1;
fi=i;

rho=10;
type1 = 7;
type2 = 2;
drho=1;
io=1;
maxIndic=-10000;
figure;
hold on;
i =1; j=1;
verbose = 1;
distpoints1=pdist2(xapp(1:14,:),xapp(1:14,:));
distpoints=pdist2(xapp,xapp);
refmed=median(median(distpoints));
% Ke = exp(- pdist2(xapp,xapp)/(2*dSigma^2)) ;
lambda = 2.5;
n_o =4;
wlambda = 0.1;
%% starting loop 
dlo=0.01;
dl= 0.1;

dSigma = 0.1*refmed;
ts=1;
while 1 % for Sigma  
    i=1;
    lambda=0.1;
    Ke = exp(- pdist2(xapp,xapp)/(2*dSigma^2)) ;
    while 1 % for lambda       
        lambda_o=0.01;
        j=1;
        while 1 % for lambda_o 
            %----------------------Calling Classifiers----------------------
            tic
            brdr=1;
            [BoxXY]=findAxis(xapp,brdr);
            [retResult,info1,info2]=clresult(Ke,type1,type2,xapp,yapp,lambda,lambda_o,wlambda,BoxXY,initL,Ke,...
                                             dSigma,n_o);
            %----------------------Saving data------------------------------
            Indivalvec=abs(info1.primalw_oxT_i);
            avgvec=sum(Indivalvec(1:14))/14;
            IndicatorValue = (sum(Indivalvec(15:18)))/(4*avgvec);
            if IndicatorValue > maxIndic
                maxIndic = IndicatorValue;
                maxlambda= lambda;
                maxlambda_o=lambda_o;
            end
            IndiMatrix(i,j)=IndicatorValue;
            
            W_oxT(i,j)   = sum(Indivalvec(15:18));
            Indilambda_o(j)=lambda_o;
            Indilambda (i) =lambda;
%%          For Showing results
            %lossVecmat(i)=retResult.lossVec;
            %----------------------Plotting------------------------------
            %outliertestfo(i,j)=info1.outliertest.f_o;
            %flo(i,j)=lambda_o;
            %outliertestf(i,j)=info1.outliertest.f;
            %fl(i,j)=lambda;
% 
%             figure;
%             %subplot(vn,hn,fi);
%             subplot(2,2,1);
% 
%             ttile='simple classifier';
%             info1.normW=0;
%             info1.normW_o=0;
%             dataContour(retResult.xtesta1,retResult.xtesta2,...
%                         retResult.ypredmat,retResult.ypred,...
%                         lambda,info1.normW,ttile,...
%                         xapp,yapp);
% 
%             %subplot(vn,hn,fi+1);
%             subplot(2,2,2);
%         %    figure;
%             ttile='Complex classifier';
%             dataContour(retResult.xtesta1,retResult.xtesta2,...
%                         retResult.ypredmat_o,retResult.ypred_o,...
%                         lambda_o,info1.normW_o,ttile,...
%                         xapp,yapp);
%             %ydiff = abs(retResult.ypredmat-retResult.ypredmat_o);
% 
%             %subplot(vn,hn,fi+2);
%             subplot(2,2,3);
%         %     figure;
%         %     ttile='Diff Classifiers';
%         %     dataContour(retResult.xtesta1,retResult.xtesta2,retResult.ypredmatf,retResult.ypred,lambda_o,info1.normW_o,ttile);
%         %     subplot(vn,hn,fi+3);
%             %subplot(2,2,4);
%             %figure;
%             ttile='Ordinary classifier';
%             info2.normW=0;
%             dataContour(retResult.xtesta1,retResult.xtesta2,...
%                         retResult.ypredmatsf,retResult.ypredsf,...
%                         lambda,info2.normW,ttile,...
%                         xapp,yapp);
%             %subplot(vn,hn,fi+3);
% 
%         %     ttile='Simple Clf-Ordinary Clf';
%         %     ydiff = retResult.ypredmat-retResult.ypredmatsf;
%         %     dataContour(retResult.xtesta1,retResult.xtesta2,ydiff,retResult.ypred,lambda_o,info1.normW_o,ttile);
%             %dataSurf(retResult.xtesta1,retResult.xtesta2,retResult.ypredmat,retResult.ypredmat_o,retResult.ypredmatsf);
% 
%             showTrPointResult(info1);
%             %fi = fi+hn;
% 
%             %rho = rho +drho;
%             %lambda_o = lambda_o-dlo;
%     %         if lambda_o<0
%     %             break;
%     %         end
% 
%%          
            lambda_o =lambda_o+dlo;
            if  lambda_o > 10 %lambda_o<0
               break;
            end
            j=j+1;
            toc
            sout=sprintf('computation  %f, %f %f',lambda,lambda_o,dSigma);
            disp(sout);
        end
        %     plot(flo(i,:),outliertestfo(i,:),flo(i,:),outliertestf(i,:));
        i=i+1;
        lambda = lambda+dl;
        if lambda > 15 
            break;
        end
    end
    fname=strcat('LambdaExp',num2str(ts));
    save(fname,'IndiMatrix','Indilambda_o','Indilambda');

    dSigma=dSigma+0.05*refmed;
    ts=ts+1;
    if dSigma > 2*refmed
        break;
    end
end
% figname=['Whyfinoutliersishigh1','.fig'];
% savefig(figname);
% for i=1:size(infoMatrix,1)
%     lambdaVec(i)  = lambda;
%     lambda_oVec(i)= lambda_o;
%     normW(i)   =info1.normW;
%     if ~info1.solsf
%         normW_o(i) =infoMatrix(i).normW_o;
%         SigAlpha(i)=infoMatrix(i).SigAlpha;
%         hMat(:,i)  =infoMatrix(i).h;
%         rMat(:,i)  =infoMatrix(i).r;
%         gMat(:,i)  =infoMatrix(i).g;
%         tr(i)      =infoMatrix(i).v;
%         AlphaMat(:,i)=infoMatrix(i).alpha_r;
%         S(:,i)=infoMatrix(i).alpha_r.*yapp+infoMatrix(i).h-infoMatrix(i).r;
%     end 
% end
% figure;
% subplot(4,2,1);
% plot(lambdaVec,normW);
% subplot(4,2,3);
% plot(lambdaVec,normW_o);
% subplot(4,2,5);
% plot(lambdaVec,SigAlpha);
% subplot(4,2,2)
% plot(lambdaVec,sqrt(normW./normW_o));
% subplot(4,2,4)
% plot(lambdaVec,normW./SigAlpha);
% subplot(4,2,6)
% plot(lambdaVec,normW_o./SigAlpha);
% subplot(4,2,7);
% plot(lambdaVec,gMat'*ONESn);
% subplot(4,2,8);
% plot(lambdaVec,S'*ONESn);


function [retResult,info1,info2]=clresult(K,type1,type2,xapp,yapp,lambda,lambda_o,wlambda,BoxXY,initL,K_o,Sigma,n_o)
    epsilon = .000001;
    kerneloption= 1;
    kernel='gaussian';
    


    [info1]=twofuncsvm(K,type1,xapp,yapp,lambda,lambda_o,wlambda,epsilon,kernel,kerneloption,rho,initL,K_o,n_o);


  
    [info2]=twofuncsvm(K,type2,xapp,yapp,lambda,lambda_o,wlambda,epsilon,kernel,kerneloption,rho,initL,K_o,n_o);

    %info2=info1;
    %% --------------Testing Generalization performance ---------------
    retResult=1;
%     reso= 0.05;
%     [xtesta1,xtesta2]=meshgrid([BoxXY.xmina:reso:BoxXY.xmaxa],[BoxXY.ymina:reso:BoxXY.ymaxa]);
%     [na,nb] =size(xtesta1);
%     xtest1=reshape(xtesta1,1,na*nb);
%     xtest2=reshape(xtesta2,1,na*nb);
%     xtest=[xtest1;xtest2]';
%     [ret_ypred] = mytwofuncsvmval(Sigma,xtest,xapp,yapp,info1,lambda,lambda_o,wlambda);
%     
%     %[lossVec] = comploss(K,xapp,yapp,info1,info2,lambda,lambda_o);
%     ypredmat  =reshape(ret_ypred.ypred,na,nb);
%     ypredmat_o=reshape(ret_ypred.ypred_o,na,nb);
%     if ret_ypred.type==4
%         ypredmatf = reshape(ret_ypred.ypredf,na,na);
%     else 
%         ypredmatf = ypredmat;
%     end
%     retResult.xtesta1  = xtesta1;
%     retResult.xtesta2  = xtesta2;
% 
%     retResult.ypredmat  =ypredmat;
%     retResult.ypredmat_o=ypredmat_o;
% 
%     retResult.ypredmatf =ypredmatf;
% 
%     retResult.ypred     = ret_ypred.ypred;
%     retResult.ypred_o   = ret_ypred.ypred_o;
%     
%     [ret_ypred2] = mytwofuncsvmval(Sigma,xtest,xapp,yapp,info2,lambda,lambda_o,wlambda);
%     %ret_ypred2=ret_ypred;
%     ypredmatsf = reshape(ret_ypred2.ypred,na,nb);
%     retResult.ypredmatsf= ypredmatsf;
%     retResult.ypredsf   = ret_ypred2.ypred;
%     %retResult.lossVec   = lossVec;
end
function dataContour(xtesta1,xtesta2,ypredmat,ypred,lambda,normW,ttile,xapp,yapp)
        ind1 =find(ypred>0);
        indm1=find(ypred<0);
        contourf(xtesta1,xtesta2,ypredmat,50);shading flat;
        hold on
        [cs,h]=contour(xtesta1,xtesta2,ypredmat,[-1 0 1],'k');
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
        showpoints(xapp,yapp);
        lambtex=sprintf('%0.5f',lambda);
        normtex =sprintf('norm W:%0.5f',normW);
        title(ttile);
        ylabel(normtex);
        xlabel(lambtex);
        
        
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