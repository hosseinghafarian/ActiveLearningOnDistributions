function MainActiveLearning01
%% Setting Solver
% Mosek
close all
addpath 'C:\Program Files\Mosek\7\toolbox\r2013a'
% YALMIP
sdpsettings('solver','mosek');
%sdpsettings('solver','sdpt3');
%yalmip('solver','sdpnal');
%sdpsettings('solver','sdpnal','verbose',2','debug',1)
% sdpsettings('debug',true);'
% CVX
% cvx_solver mosek
% cvx_save_prefs
%% Active learning method: Active learning Method 
%        1:paper combining active learning...gaussian random fields
%        2:paper active learning using informative ... 
%% Classifier     : classification method
%  nQueryEachTime : number of queries in each call of active learner
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Global variables 
%   
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% global fig;
global prehlines;
global hpre;
global qSDP; % for returning active learning query points between learner and active learner
global batchSize;
global n_o;
global lnoiseper;%percent of labels are noisy
global onoiseper;%percent of unlabeled data are outlier
global Warmstart;
global WMST_beforeStart ;
global WMST_appendDataInd;
Warmstart = true;
WMST_beforeStart = true;
WMST_appendDataInd = [];

hpre     =[];
prehlines=[];

runprofile = 1 ; % if I want to just experiment 2dsample and see the result 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initializations
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
qSDP     = 0;        % No data is selected at first
batchSize= 1;
isDistData= false;   % Is it ditributional data or vectorial data?

%Lindex = [];         
%Uindex = [];
%Tindex = [];
%Allindex = [];
InitL  = [];
Startinglambda =1;deltalambda=0;% starting value for lambda and increase in lambda every time
numberOfLabeled = size(InitL);  % number of labeled data
TestRatio    = 0.01;            % what percentage of data will be used for test?
SampleRatio  = 0.02;            % what percentage of data will be used for train?
%% Active Learning Method 
%  1: Random sampling : use it with SVM
%  2: Quire    : querying informative and representative samples  : use it
%  with SVM
%  3: No Active learning, : use all data
%  4: My SDP AL
%  5: My Diverse Subtask AL
%  6: 
%  7: 
%  8: 
%  9: Uncertainty Sampling 
% 10: 
% ALmethod    = 9;
% batchSize   = 1;
% numberOfInitL = 2;              % How many data points for the start?
AccuracyStoreIndex= {[1],[2],[],[],[6,7,8]',[6,7,8]',[],[],[9],[10]};
% list of noise resistant methods
noiseResMethods = [6];
    %% Initial Samples Selection
    % How to initialy select random samples? 
    % 1: Select Two random samples
    % 2: Select Two random samples from different classes
    % 3: All of samples (Passiver learning)
    % 4: select sample by user.
    initType  = 2; 
%%
%  1: SVM
%  2: Least Squares
%  3: My SDP formulation: Convex Relaxation through Semi-definite Programming Using Alternating Minimzation
%  4: My Diverse subTask
%  5: Active Direct Convex Relax
%  6: Active Convex Relax outlier and Label noise robust
%  7: 
%classifier= 6; % TODO: Some CLassifier methods are also AL methods, solve conflict between 
               %       classifier and ALmethods    
%% Options for kernel
KOptions.KernelType = 2; % 1: Linear, 2: RBF, 
KOptions.Sigma2     = 1; %Kernel function bandwidth Sqaure, e^(-\Vert x-y\Vert^2/(2*Sigma2))for Input Space
KOptions.SigmaInInputSpace = 0.5;% TODO: Verify Kernel function for Embedding Space
%% Options for 2D Sample plot
%lOptions.threshold = 14;
%lOptions.lowBValue = -1;
%lOptions.highBValue= 1;
showData   = false ;    % To show data in 2d experiments or not?
show2DData = false;
viewTransformSpace = true;
showOptions.clear=true;
showOptions.acceptNewInstance=false;
showOptions.selectInstance = false;
showOptions.StopforKey     = false;
showOptions.isUpdate = false;
showOptions.KOptions = KOptions;
showOptions.showContour = false;
showOptions.model= 0;
showOptions.showIndex   = false;
figName='Samples';
%% Options for Learning and AL

SynthesisData = false; % data is synthesised or from a dataset file
Options.Transductive  = false;     % Transductive or not? 
if runprofile == 1 % If want to do 2dsample test 
    SynthesisData = true;
    Options.Transductive  = true;     % Transductive or not? 
    showData   = true ;               % To show data in 2d experiments or not?
    show2DData = true;
    TestRatio    = 1;            % what percentage of data will be used for test?
    SampleRatio  = 1;            % what percentage of data will be used for train?
end
%% Load data file(s)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  load data file(s)
%  data is loaded from datafile and using cross validate some part of data 
%  selected for train. Active learning will select some part of train data
%  The resulting classifier will be tested on Test set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dataset Id 
if SynthesisData
    startdataset = 5;
    enddataset  = 10;
    datasetName = 'synthesis';
else 
    startdataset  = 5;
    enddataset    = 9;
end
almethod_num  = [9];%,6,2,5,9,10];% almethods and corresponding classifiers. 
classifer_num = [9];%,6,1,5,1, 1];% almethods and corresponding classifiers.
max_method_ind = 6;
max_batchsize  = 5;
datasetId  = startdataset;
%% Experiment On Data Set
while datasetId <= enddataset 
    %% Load Data Set
    if SynthesisData
        %% Synthesis Data 
        %datafile ='tile2to2.mat';
        N        = 100;
        %datafile='linearUnlabeledsvm.mat';
        %datafile='orderedlittle.mat';
        %datafile='orderedlittleOutlier2t2copy.mat';
        datafile ='Copy_of_outlier4orderedlittle.mat';
        %datafile='outlier4orderedlittle.mat';
        %datafile='Outlier_orderedlittle.mat';
        %datafile='orderedlittleOutlier2t1.mat';
        reWriteData   = 0;
        if ~exist(datafile, 'file') || reWriteData
            %[X,Z,TS] = getRandomData(isDistData,N,50);
            %save(datafile,'X','Z','TS');
            XClass = sampleShow(figName);
            save(datafile,'XClass');
        else
            load(datafile);
        end
        XClass = XClass';
        n      = size(XClass,2);
        X      = [XClass(1:2,:);ones(1,n)];
        Z      = XClass(3,:);
        
        % move samples 1,2 for checking correctness of initL,etc
        
        RX = X ( :,1:2);
        ZX = Z (1:2);
        
        X(:,1:2) = X(:,[5,9]);
        Z(1:2)   = Z(:,[5,9]);
        X(:,[5,9]) = RX;
        Z([5,9])   = ZX;
        numberOfLabeled=0;
        strnameext = strsplit(datafile,'.');
        datasetName = strcat('syth-',strnameext{1});
    else
        %% Data Set 
        %% Data set ID: TODO: Writing some of dataset loading code 
        % 1 : Breast Cancer Dataset
        % 2 : Bupa Liver Dataset
        % 3 : Isolet 1+2+3+4
        % 4 : isolet 5
        % 5 : letter  'E', 'D', 6-9 other letter data set loading exactly same
        % 10: MNIST data set
        % 11: Vehicle ,UCI
        % 12: ionosphere ,UCI
        %% Data set Load
        [data,datasetName]    = loaddataset(datasetId);
        XClass    = data.X;
        Z         = data.Y';
        X         = XClass';
    end
    % paramerters for adding noise to data 
    noisetype    =  1; % 0: add no noise, 1: label noise 
    noisepercent = 10;
    query_round_num = 20;
    exp_num   = 10;
    method_num = 5;
    maxbatchsize    = 5;
    assert(batchSize <=maxbatchsize);
    %  complexitych   = [complexitych  ; ModelAndData.lambda];
    initL          = cell(10,query_round_num);  
    for i=1:10
        for j=1:query_round_num
            initL{i,j}= zeros(maxbatchsize,1); 
        end
    end
    accuracy     = zeros(query_round_num,exp_num,method_num);
    sizeinitLset = zeros(query_round_num,exp_num,method_num);

    method_ind     = 1; 
    %%  Method ( Active Learning Method)   
    for method_ind=1:method_num
        % setup active learning and classifier method, batchsize and etc. 
        ALmethod      = almethod_num(method_ind);
        classifier    = classifer_num(method_ind);
        Options.Querymethod   = ALmethod ; % Active Learning Method
        Options.classifier    = classifier;% Classifier Method 
        Options.nQueryEachTime= batchSize; % Batch Size 
        Options.KOptions      = KOptions;  % Options for Kernel
        %
        batchSize     = 1;
        numberOfInitL = 2;     % How many data points for the start?
        % set file name for experiments
        expdatafile   = makexperimentFilename(ALmethod,datasetId,datasetName,noisetype,noisepercent);
        repeatexp     = 1;
        %% Repeat Experiments to compute average performance
        for repeatexp=1:exp_num 
            %% Select Train and Test samples, and Compute Kernel 
            % Transpose data to make every data columnize
            % And Append one to every x instead of b in w^Tx+b
            % Select Test indices
            reComp = 1;%reWriteData; %  1: compute Kernel, 0: use previous saved file 
            kernelcfile='TTKdLUSVM03.mat'; % save file for Kernel,TrainSamples and TestSamples
    %         if ~exist(kernelcfile, 'file') || reComp
                seltype = 1;
                [TrainSamples,TestSamples]= selectTrainAndTestSamples(seltype,X,Z,TestRatio,SampleRatio,Options.Transductive);
                K = KernelMatrix(isDistData,TrainSamples.xTrain,KOptions);
    %             save(kernelcfile,'K','TrainSamples','TestSamples');
    %         else
    %             load(kernelcfile);
    %         end 
            xTrain = TrainSamples.xTrain ;
            yTrain = TrainSamples.yTrain ;
            %% Noise Addition to data for test
            % add noise to data     
            n_o = 0; % assume no data is noisy 
            n   = size(yTrain,1);
            % the following two lines means we didn't make them noisy. 
            isnoisy   = false(n,1);
            isoutlier = false(n,1);
            if false%ismember(ALmethod,noiseResMethods)
                [xTrain,yTrain,isnoisy,isoutlier,n_o,lnoiseper,onoiseper] = makedatanoisy(xTrain, yTrain, noisepercent, noisetype);
                onoiseper = (4/18)*100;
                n_o = 4;%round(0.05*n);
            end
            xTest  = TestSamples.xTest;
            yTest  = TestSamples.yTest;
            yzTest = zeros(1,size(yTest,1));
            KA     = KernelArray(isDistData,xTrain,xTest,KOptions,1);% Kernel between train and test samples,last parameter is recomp
            %K=getKernelMatrix(xTrain,n,KOptions,reCompKernel);
            %% Model and Data: Kernel Matrix, lambda , C  
            n      = size(xTrain,2); % number of train data points  
            lambda = Startinglambda; % lambda: classifier complexity
            ModelAndData.TrSize = n; % Size of the training data 
            ModelAndData.Kernel = K;
            ModelAndData.lambda = lambda;
            ModelAndData.model  = 0; % only to remember we have model 
            ModelAndData.KA     = KA;
            ModelAndData.isnoisy   = isnoisy;
            ModelAndData.isoutlier = isoutlier;
            %% Select Initial Active Learning samples
            if SynthesisData
                %% Synthesised Data, just select data points 1 and 2 
                initL{repeatexp}   = [5,9]';%[1;2];
                if Options.Transductive 
                    XSh = [xTrain];
                    YSh = [yTrain];
                    sampleShow(figName,XSh,YSh',initL{repeatexp},showOptions);
                else
                    XSh = [xTrain,xTest];
                    YSh = [yTrain,yzTest];
                    sampleShow(figName,XSh,YSh',initL{repeatexp},showOptions);
                end
            else 
                %% Select initial samples 
                initL{repeatexp,1} = initialSamplesSelected(initType,xTrain, yTrain);
            end
            %% Initialize main active learning loop, Call Learner And Active Learner Before Main Active Learning loop
            queryInstance = 0;                
            initLcurr   = zeros(batchSize*query_round_num,1);
            initLcurr(1:size(initL{repeatexp,1},1)) = initL{repeatexp,1};
            % Calling Learner 
            WMST_beforeStart = true;
            [acc_ML, empRisk,Predict,model]...
                  = Learner(isDistData,Options,ModelAndData,initL{repeatexp,1},...
                            xTrain, yTrain, ...
                            xTest,  yTest ,queryInstance   );
            % Calling Active Learner ( Query Selection )
            [queryInstance]...
                  = ActiveLearner(isDistData,Options,ModelAndData,initL{repeatexp,1},...
                                xTrain, yTrain, ...
                                xTest,  yTest     );
            ModelAndData.model=model;            
            show_Data(showData,showOptions,figName,xTrain,yTrain,initL{repeatexp});
            %% Store Performance Results    
            query_round_numi = 1;
            ntotalQueries             = size(initL{repeatexp,1},1);
            accstoreidx               = AccuracyStoreIndex{ALmethod};
            accuracy(query_round_numi,repeatexp,accstoreidx) = acc_ML;
            sizeinitLset(query_round_numi,repeatexp,accstoreidx) = size(initL{repeatexp,1},1);
            initL{repeatexp,2}        = queryInstance;
            initLcurrnexInd  = size(initL{repeatexp,1},1)+1;
            initLcurr(initLcurrnexInd:initLcurrnexInd+batchSize-1) = queryInstance;
            initLcurrnexInd =initLcurrnexInd+batchSize;
            risksall     = empRisk;
%             ntotalQueries= batchSize+1  ;
            complexitych = ModelAndData.lambda;
            listOfRisks  = empRisk;
            PredSequence = Predict;
            
            %% Main Active Learning Loop
            for query_round_numi=2:query_round_num    % while ntotalQueries+batchSize < budget/3 
                %% Calling Learner 
                [acc_ML, empRisk,Predict,model]...
                      = Learner(isDistData,Options,ModelAndData,initLcurr,...
                                xTrain, yTrain, ...
                                xTest,  yTest , queryInstance    );
                %% Calling Active Learner ( Query Selection )
                [queryInstance]...
                      = ActiveLearner(isDistData,Options,ModelAndData,initLcurr,...
                                xTrain, yTrain, ...
                                xTest,  yTest      );
                ModelAndData.model=model; 
                %% Store Performance Results
                % which means that it is miscalculated, it must be checked.
                ntotalQueries                         = size(initL{repeatexp,1},1);
                accuracy(query_round_numi,repeatexp,accstoreidx)     = acc_ML;
                sizeinitLset(query_round_numi,repeatexp,accstoreidx) = size(initL{repeatexp},1);
%                 
%                 accuracy{repeatexp,1}       = [accuracy{repeatexp},       acc_ML];
%                 sizeinitLset{repeatexp,1}   = [sizeinitLset{repeatexp}  ,size(initL{repeatexp},1)];
%                 %complexitych   = [complexitych  ; ModelAndData.lambda];
                initL{repeatexp,query_round_numi+1}       = queryInstance;
                initLcurr(initLcurrnexInd:initLcurrnexInd+batchSize-1) = queryInstance;
                initLcurrnexInd =initLcurrnexInd+batchSize;
                %listOfRisks    = [listOfRisks   ; empRisk ]; 
                acc_ML
                ntotalQueries  =  ntotalQueries+batchSize;
                %PredSequence   = [PredSequence  ; Predict ];
                show_Data(showData,showOptions,figName,xTrain,yTrain,initL{repeatexp});
                %% Update Model and Data
                % we must have an intelligent method for updating lambda
                % first we have to know whether the label of query instance successfully
                % predicted or not?
                % if label of queryInstance correctly predicted, it means that 
                % may be classifier is complex enough
    %             if Predict 
    % 
    %             else
    % 
    %             end
    %             ModelAndData.lambda = ModelAndData.lambda-deltalambda;
            end
            save(expdatafile,'accuracy','sizeinitLset','initL');
            repeatexp = repeatexp + 1;
        end % repeat experiment
        ACC_PLOT_Data = computeAverage(accuracy,sizeinitLset,initL,exp_num,method_num,query_round_num);
        Plotfilename = strcat('PlotData_DS_',datasetName);
        Plotfilename = strcat(Plotfilename, '.mat');
        save(Plotfilename,'ACC_PLOT_Data');
    end
end % dataset     
%% Save Results

%% Show Results, and Save them
% figure;
% plot(accuracy);
%% 
% it is a very good test to increase only lambda and wait to see when
% empRisk start to decrease
% i have to write a code to store results in a meaningful way. 
ResultFile = 'resultfile.mat';
save(ResultFile,'accuracy','complexitych','listOfRisks','PredSequence');
end
function ACC_PLOT_Data = computeAverage(accuracy,sizeinitLset,initL,exp_num,method_num,query_num)
    ACC_PLOT_Data = zeros(method_num,query_num);
    for i=1:method_num
        ACC_PLOT_Data(i,:)= 0;
        varexp_num = zeros(1,query_num);
        for j=1:exp_num
            acc = reshape(accuracy(:,j,i),1,query_num);
            %varexp_num  = varexp_num + acc~=0;
            ACC_PLOT_Data(i,:) = ACC_PLOT_Data(i,:) + acc;
        end
        ACC_PLOT_Data(i,:)= ACC_PLOT_Data(i,:) / exp_num;
    end
end
function show_Data(showData,showOptions,figName,xTrain,yTrain,initLexp)
    if showData
        showOptions.showContour = true;
        %showOptions.model       = ModelAndData.model;
        showOptions.showIndex   = true;
        showOptions.isUpdate    = true;
        sampleShow(figName,xTrain,yTrain',initLexp,showOptions);
    end
end
function fname = makexperimentFilename(ALmethod,datasetId,datasetName,noisetype,noisepercent)
        strnoise    = strcat('noiseT',num2str(noisetype));
        strnoise    = strcat(strnoise,'-noisePer');
        strnoise    = strcat(strnoise,num2str(noisepercent));
        strmethod   = strcat('methodnum-',num2str(ALmethod));
        expdatafile = strcat('expDS-',num2str(datasetId));
        expdatafile = strcat(expdatafile, datasetName);
        expdatafile = strcat(expdatafile, strmethod);
        expdatafile = strcat(expdatafile, strnoise);
        fname       = strcat(expdatafile, '.mat');
end