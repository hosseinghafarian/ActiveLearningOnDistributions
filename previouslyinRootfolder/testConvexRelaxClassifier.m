function [predict_label, accuracy, f_test] = testConvexRelaxClassifier(x_labeled, y_labeled, xtest, ytest, y_transductive, model_set, Y_set, learningparams)
% This function must be checked. 
global cnstData
%% Compute Resulting Classifier accuracy
% TODO : G in this form is not appropriate since it has q in it. if the last q_i actively selected is a good 
%        choice then it is large for that and then G makes it's effect
%        small on the learning. 
%        Two ways to avoid this : 1-simple method:estimate Y for unlabeled data and then use YY' instead of G 
%                                 2-more involved:learn semisupervised
%                                 learning of the function with the new
%                                 labeled ( with p and without q) 
    function [y_predtest, alpha_pv, valid_classifier,accuracy_cvxRelaxclassifier] = relearnclassifier ()
        global cnstDefs        
        alpha_p = sdpvar(n_S,1);
        cConstraint= [alpha_p>=0,alpha_p<=1];
        G          = proj_sdp(model_set.G(1:n_S,1:n_S),n_S);
        cObjective =-sum(alpha_p.*model_set.h(1:n_S))...
                +1/(2*lambda)* alpha_p'*(cnstData.K.*G )*alpha_p;%+2/lambda* alpha_p'*(cnstData.K.*G )*alpha_p;
        opts = sdpsettings('verbose',cnstDefs.solver_verbose, 'solver','mosek');
        sol=optimize(cConstraint, cObjective, opts);
        if sol.problem==0
            valid_classifier = true;
            alpha_pv=value(alpha_p);                             
%             setnon_noisy = setdiff(cnstData.unlabeled, model_set.noisySamplesInd);
            Kerneltest = cnstData.KA(1:cnstData.n_S,:);% [initL',unlabeled'],:);
            %Y_D = zeros(cnstData.n_S,1);
            assert(norm(Y_set(cnstData.initL(cnstData.initLnozero))' - cnstData.Yl(cnstData.initL(cnstData.initLnozero)))==0);
            %Y_D = Y_set;
            dfn = norm(alpha_pv-model_set.alpha_pv(1:n_S));
            f_test  = Kerneltest'*(alpha_pv.*model_set.h(1:n_S).*Y_set);
            y_predtest = sign(f_test)';
            niseq        = bsxfun(@eq, y_predtest, ytest);
            niseq        = sum(niseq);
            accuracy_cvxRelaxclassifier= niseq/numel(y_predtest)*100;
        else
            alpha_pv         = zeros(n_S,1);
            y_predtest       = zeros(numel(ytest),1);
            valid_classifier = false;
            accuracy_cvxRelaxclassifier = -1;
        end
    end
    function [y_predtest, alpha_pv, f_test, valid_classifier,accuracy_cvxRelaxclassifier] = inherent_classifier()
        valid_classifier = true;
%         setnon_noisy     = setdiff(cnstData.unlabeled, model_set.noisySamplesInd);
        alpha_pv         = model_set.alpha_pv;
        Kerneltest       = cnstData.KA(1:cnstData.n_S,:);% [initL',unlabeled'],:);
        f_test           = Kerneltest'*(model_set.alpha_pv(1:n_S).*model_set.h(1:n_S).*Y_set);
        y_predtest       = sign(f_test);
        niseq2           = bsxfun(@eq, y_predtest, ytest);
        niseq2           = sum(niseq2);
        accuracy_cvxRelaxclassifier = niseq2/size(y_predtest,1)*100;
    end
    function [trans_accuracy] = compute_transductive_accuracy(YfCheck) 
        Yun            = sign ( Y_set );
        niseq          = bsxfun(@eq,Yun', YfCheck);
        niseq          = sum(niseq);
        trans_accuracy = niseq/n_S*100;        
    end
%% Begin
n_S     = cnstData.n_S;
n_u     = cnstData.n_u;
unlab   = cnstData.unlabeled;
lambda  = learningparams.lambda;

which_cl  = 2;
if which_cl==1
    [predict_label, ~, f_test, valid_classifier, accuracy_cvxRelaxclassifier]   = inherent_classifier();
elseif which_cl == 2 
    [predict_label, f_test, valid_classifier, accuracy_cvxRelaxclassifier]   = relearnclassifier ();
else
    [predict_label, ~, f_test, valid_classifier_i,accuracy_cvxRelaxclassifier] = inherent_classifier();
    [y_predtest, ~, f_test, valid_classifier,~] = relearnclassifier ();
    diff_pred = norm(y_predtest-predict_label);
end    
%% Computing Accuracy of SVM
%          TODO: labels of semisupervised learning method, is not used for learning with svm, 
%                   1- learning with just labeled samples
%                   2- learning with labeled samples and unlabeled samples using labels predicted by classifier 
% nolabelnoise_initL = setdiff(initL,model_set.noisySamplesInd);

[predict_label, accuracy_lib]=libsvm_trainandtest(x_labeled, y_labeled', xtest(:,:), ytest', learningparams);
if exist('accuracy_lib','var')
    accuracy_svm = accuracy_lib(1);
else
    accuracy_svm = -1;
end
%% computing Transductive accuracy  
    % 1- we must not compute outlier/label noisy samples in
    % transductive accuracy
    % 2- may be in non-noisy datasets we must decrease n_o as we move
    % forward
    accuracy_transductive = compute_transductive_accuracy(y_transductive);
%% setting returning values
accuracy(3) = accuracy_cvxRelaxclassifier;
accuracy(2) = accuracy_transductive;
accuracy(1) = accuracy_svm;
end