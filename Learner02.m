function [acc_Result, empRisk,Predict,model]...
            = Learner02( isDistData,Options, ModelInfo ,learningparams,initL,...
                             xTrain,     yTrain,    ...
                             xTest,      yTest ,queryInstance     )
%% Main Learning function
%% Initialize    Variables 
global qSDP;        %for setting active learning result by SDP Method
global batchSize;   % batchSize for batch active learning
global n_o;
global lnoiseper;
global onoiseper;
global cnstData;

empRisk       = 0; % this is nothing except for LEAST SQUARES
Predict       = 0; % this is nothing except for LEAST SQUARES

method        = Options.Querymethod;
classifier    = Options.classifier;
nQueryEachTime= Options.batchSize;% batch size
transductive  = Options.Transductive;  % Learning and active learning on the same set
lambda        = learningparams.lambda;
model.dv =0;
%% Initialize    Classifier
sw = 1;
if classifier==4  % if diverse subtask active learning
    %% Diverse subTask SDP Active Learning
    %  use method 2 of the same function as classifer 3 
    sw = 2;         % use method 2 
    classifier = 3; % of the same function as classifier= 3
end
%% Calling Train Classifier
switch classifier % just for cases 1, 2, 3
%% FirstMethods
    case 1 % SVM
        [model]                                    = svmtrainwrapper(warmStart, x_primalwmst, y_dualwmst, alpha_primalwmst, ModelInfo, learningparams);
        [predict_label, accuracy, decision_values] = svmpredictwrapper(xtest, ytest, model);
        model.dv = decision_values; 
    case 2 % Least Sqaures 
        [model] = leastsquareclassifier(warmStart, x_primalwmst, y_dualwmst, alpha_primalwmst, Model, learningparams);
    case 3 % Convex Relaxation through Semi-definite Programming Using Alternating Minimzation
        % case 4 is diverse subtask see above 
        [qSDP, q, model, Y_set]= ActiveDiverseMultitask(sw,initL,Yl,lambda);
    case 4
        %% Doing Active minimax algorithm using unlabeled and labeled data
        [qSDP, q,model_set,Y_set]= ActiveCRSemiSVM03(sw,initL,Yl,learningparams.lambda);
        model.model = model_set;
        model.type  = 4;               % 4 : diverse subtask
        model.T     = size(model_set); % T : if diverse subtask
        %% Setting Active Learning Results 
        thresholdQ = 0.9;
        [~,ql]=max(q);
        %ql = (q > thresholdQ);
        qSDP = unlabeled(ql);  
          
    case {5,6,7,8,9}
%% Second Methods
        %% Finding labeled, unlabeled and query data indices
        unlabeled    = cnstData.unlabeled;
        if numel(unlabeled)==0 
            return 
        end
        %% Make Kernel matrices of labeled,unlabeled and query data        
        Yl  = yTrain(initL');
        % This code minimizes convex relaxation of transductive svm 
        %[g,Z]= SemiSVMConvexRelax(Kll,Klu,Kuu,Yl,n_l,n_u,lambda)
        switch classifier
            case 5
                 % The following methods is equivalent and it is tested on some dataset.
                 % In the first method we have beta_p>=0 and eta_p>=0, but in the
                 % second method, u=beta_p-eta_p and we used norm one and
                 % hinge to have the same objective function. 
                 [ALresult,model_set,Y_set]= ActiveDConvexRelax2(warmStart, x_primalwmst, y_dualwmst, alpha_primalwmst, ModelInfo, learningparams);   
%                [ALresult,model_set,Y_set]= ActiveDConvexRelax3(sw,ModelAndData,initL,unlabeled,Yl,n_l,n_u,batchSize,lambda);
            case 6 % TODO: Change Code to the form of case 5, Change cases 5, 6 to the consider WARMSTART
                % This code minimizes convex relaxation of transductive svm 
                %[g,Z]= SemiSVMConvexRelax(Kll,Klu,Kuu,Yl,n_l,n_u,lambda)
                %% Convex Relax Active Learning Outlier
                [ALresult,model_set,Y_set] = ActiveDConvexRelaxOutlier(warmStart, x_primalwmst, y_dualwmst, alpha_primalwmst, ModelInfo, learningparams);          
            case 7
                %% Convex Relax Active Learning 
                %[g,Z]= ActiveCRSemiSVM(Kll,Klu,Kuu,Yl,n_l,n_u,lambda)
%                 batchSize = 1;
                %[q,model_set,Y_set]= ActiveDConvexRelaxOutlierEM(sw,Kll,Klu,Kuu,Yl,n_l,n_u,batchSize,lambda);
            case 8
                [qresult,model_set,Y_set]= ActiveDConvexRelaxOutlierProxADMM2(warmStart, x_primalwmst, y_dualwmst, alpha_primalwmst, ModelInfo, learningparams);
                tq = k_mostlargest(qresult,batchSize);
                qi = unlabeled(tq); 
                ALresult.queryInd = qi;
            case 9
                saved = true;
                if ~saved
                     [ALresult,model_set,Y_set,x_opt,alpha_opt,obj_opt] = ActiveDConvexRelaxOutlier(sw,ModelInfo,learningparams,initL,cnstData.unlabeled,Yl,cnstData.n_l,cnstData.n_u,batchSize,n_o,lnoiseper,onoiseper);          
                     ModelInfo.ALresultComp  = ALresult;
                     ModelInfo.model_setComp = model_set;
                     ModelInfo.Y_set1        = Y_set;              
                     save('solverresultdata','ModelInfo','x_opt','alpha_opt','obj_opt');
                 else
                     load('solverresultdata','ModelInfo','x_opt','alpha_opt','obj_opt');
                 end; 
                 warmStart         = false;
                 x_primalwmst      = 0;
                 y_dualwmst        = 0;
                 alpha_primalwmst  = 0;
                 %%[ALresult,model_set,Y_set] = ActiveOutlierSaddle11(sw,ModelInfo,learningparams,initL,unlabeled,Yl,cnstData.n_l,cnstData.n_u,cnstData.batchSize,n_o,lnoiseper,onoiseper);
                 %[ALresult, model_set, Y_set] = ActiveOutlierSaddle12(warmStart,x_primalwmst,y_dualwmst,ModelInfo,learningparams);   
                 %[ALresult, model_set, Y_set] = ActiveOutlierSaddle13(warmStart,x_primalwmst,y_dualwmst,ModelInfo,learningparams);   
                 %[ALresult, model_set, Y_set] = ActiveOutlierSaddle14(warmStart,x_primalwmst,y_dualwmst,ModelInfo,learningparams);
                 [ALresult, model_set, Y_set] = ActiveOutlierSaddle15(warmStart,x_primalwmst,y_dualwmst,alpha_primalwmst,ModelInfo,learningparams,x_opt,alpha_opt,obj_opt);   
                 % For calling ActiveOutlierSaddle1-10 goto MainActiveLearner01.m and Learner.m 
                 ModelInfo.Kernel  = Ker;
        end
        yalmip('solver','mosek');
        model.model = model_set;
        model.type  = 4;               % 4 : diverse subtask
        model.T     = size(model_set); % T : if diverse subtask
        %% Setting Active Learning Results 
%         thresholdQ = 0.9;
%         [~,ql]=max(q);
        %ql = (q > thresholdQ);
        qSDP = ALresult.queryInd;             
end     
%% Calling Test  Classifier
switch classifier 
    case {5,6}
        %% TODO: This part needs carefull consideration, it seems that results are incorrect.
        %        Furthermore, when number of unlabeled data is low, by querying
        %        and therefore reducing number of unlabeled data, accuracy
        %        decreases because denominator decreases and even number of incorrect
        %        classified samples decreases, accuracy decreases or not
        %        increases much. 
        %% Compute Accuracies :  1-Accuracy with Resulting classifier, 2-Transductive Accuracy, 3- Libsvm, not semisupervised Accuracy
        acc_Result = zeros(3,1);
        [alpha_pv  , acc_Result(1)] = compAccResultingClassifier();
        [ acc_Result(2)]            = compAccTransductive(Y_set, yTrain(cnstData.unlabeled), classifier);
        [y_predtest, acc_Result(3)] = compAccSVM();
        return    
    case 1 
        % comp directly
        %         gamma= Options.KOptions.Sigma;
        %         K_test = exp(- gamma * pdist2(xTrain(:,model.svind), xTest)) ;
        %         f_test = sign(model.alphay(model.svind)' * K_test + model.b) ;
        %         niseq = bsxfun(@eq, f_test, yTest);
        %         niseq=sum(niseq);
        %% SVM
        acc_Result = accuracy(1);    
    case 2 
        %% Least Squares loss function 
        [predict_label, accuracy, decision_values] = leastsquaretest(xtest, ytest, model);
        %[acc_Result]             = compLeastSquaresLossClassifier();
    case 3 
        %% computing Transductive accuracy 
        Yun= sign ( Y_set );
        niseq = bsxfun(@eq,Yun, YfCheck);
        niseq=sum(niseq);
        acc_Result = niseq/size(YfCheck,1);
end
    function [alpha_pv,acc_Result]    = compAccResultingClassifier()
        %% TODO : for test data, is test data is outlier based on w_o function?
    %           if it is then it must be said that it is outlier. 
        sdpsettings('solver','mosek');%scs');
        alpha_p = sdpvar(cnstData.n_S,1);
        cConstraint= [alpha_p>=0,alpha_p<=1];
        cObjective =-sum(alpha_p.*model_set.g(1:cnstData.n_S))...
                    +2/lambda* alpha_p'*(cnstData.K.*model_set.G(1:cnstData.n_S,1:cnstData.n_S))*alpha_p;
        sol=optimize(cConstraint,cObjective);
        if sol.problem==0
            alpha_pv=value(alpha_p);                             
            sdpsettings('solver','scs');
            setnon_noisy = setdiff(unlabeled,model_set.noisySamplesInd);
            Kerneltest = ModelInfo.KA(1:cnstData.n_S,:);% [initL',unlabeled'],:);
            Y_D = zeros(cnstData.n_S,1);
            Y_D(initL) = Yl;
            Y_D(unlabeled) = Y_set;
            f_test = Kerneltest'*(alpha_pv.*model_set.g(1:cnstData.n_S).*Y_D);
            y_predtest = sign(f_test);
            if classifier==6 
                niseq        = bsxfun(@eq, y_predtest, yTest);
                niseq        = sum(niseq);
                acc_Result= niseq/size(y_predtest,1);
            else
                niseq    = bsxfun(@eq, y_predtest, yTest);
                niseq    = sum(niseq);
                acc_Result = niseq/size(yTest,1);
            end
        else
            acc_Result = 0;
        end
    end
    function [y_predtest, acc_Result] = compAccSVM()
        %%          TODO: labels of semisupervised learning method, is not used for learning with svm, 
%                   1- learning with just labeled samples
%                   2- learning with labeled samples and unlabeled samples using labels predicted by classifier 
        nolabelnoise_initL = setdiff(initL,model_set.noisySamplesInd);
        if classifier==6 
           [predict_label, accuracy,decision_values]=libsvm_trainandtest(xTrain(:,nolabelnoise_initL),yTrain(nolabelnoise_initL),xTest,yTest,lambda);
        else
           [predict_label, accuracy,decision_values]=libsvm_trainandtest(xTrain(:,nolabelnoise_initL),yTrain(nolabelnoise_initL),xTest,yTest,lambda);
        end
        %[predict_label, accuracy]=libsvm_trainandtest(xTrain(:,initL),yTrain(initL),xTest(:,:),yTest(:),lambda);
        if ~isempty(accuracy)
            acc_Result = accuracy(1);
        else
            acc_Result = -1;
        end
        y_predtest = predict_label;
    end
    function [acc_Result]             = compAccTransductive(Y_set, YfCheck, classifier)
        % computing Transductive accuracy  
            % 1- we must not compute outlier/label noisy samples in
            % transductive accuracy
            % 2- may be in non-noisy datasets we must decrease n_o as we move
            % forward
        Yun= sign ( Y_set );
        if classifier==6 
            %Yun = Yun(setnon_noisy);
            YfCheckc = YfCheck;
        else
            YfCheckc = YfCheck;
        end
        niseq = bsxfun(@eq,Yun, YfCheckc);
        niseq = sum(niseq);
        acc_Result = niseq/size(YfCheckc,1);    
    end
    function [acc_Result]             = compLeastSquaresLossClassifier()
        labelX=yTrain(initL);
        % Computing J(f,lambda) for labeled training data
        empRisk = labelX'*Klambdainv*labelX;
        % Does classifer correctly predicts queryInstance Label?
        KAq = KernelArrayNewInstance(isDistData,xTrain,xTrain(queryInstance),Options.KOptions);
        f=labelX'*Klambdainv*KAq{1}([initL]);
        Sf=sign(f);
        if (Sf==yTrain(queryInstance))
            Predict = true;
        else
            Predict = false;
        end
        Sf=size(yTest,1); 
        niseq=0;
        for it=1:size(xTest)
            KAl=ModelInfo.KA{it}(initL);
            f=labelX'*Klambdainv*KAl;
            Sf(t)=sign(f);
            niseq=niseq+isequal(sign(f),yTest(it));
        end
        acc_Result = niseq/size(xTest,1);  
    end
end