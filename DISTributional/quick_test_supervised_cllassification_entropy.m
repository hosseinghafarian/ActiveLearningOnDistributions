
%clear start:
    clear all; close all;
% regenerate dataset 
    regen_data  = true;
    
%parameters:    
    dataset = 'entropy'; %fixed
    %sample numbers:
        L = 100; %number of distributions
        N = 500; %number of samples/distribution
    %training, validation, test sample sizes (assumption: L = L_train + L_val + L_test):
        L_train = 25;  %number of training distributions from L
        L_val = 25;    %number of validation distributions from L
        L_test = 50;   %number of test distributions from L
    %embedding:        
        cost_name = 'expected'; %name of the used embedding; cost_name = 'expected' <-> mean embedding
        kernel =  'RBF';
        %kernel parameter:
            base_kp = 10; %fine-grainedness of the parameter scan; smaller base_kp = finer scan, 'kp': kernel parameter
            %kp_v = base_kp.^[-4:3]; %candidate kernel parameters ('RBF widths'); '_v': vector
            kp_v = base_kp.^[-1];
    %regularization parameter (lambda):
        base_rp = 10;  %fine-grainedness of the parameter scan; smaller base_rp = finer scan; 'rp': regularization parameter
        rp_v = base_rp.^[-10:-1]; %candidate regularization parameters (lambda)
    %v_v = [1:1]; %[1:25]; indices of the random runs; number of random runs = length(v_v); '_v': vector
    v_v = [1:20]; % each time a query 
    
%matlabpool open; %use this line with 'Matlab: parallel computing'
%-------------------------------
%create saving directory (if it does not exist) for the precomputed Gram matrices:
% dir_G = strcat(dataset,'_Gram'); %directory name for the precomputed Gram matrices.
% dir_present = create_and_cd_dir(dir_G);
   
objective_values = zeros(length(v_v),1);    
%for v = v_v
    
    %generate dataset:    
        [X,Y,idx_train,idx_val,idx_test,X_parameter,Z] = generate_supervised_entropy_dataset_classification(L,N,L_train,L_val,L_test);
        idx_unlabel = idx_train ;
        query_init  = randperm(L_train,3); % ranodmly choosing the first labeled points
        idx_train   = idx_unlabel(query_init);
        idx_unlabel = setdiff(idx_unlabel,idx_train); % updating unlabel points
        L_train_all = L_train;
        L_train = size(idx_train,2);
v=1;        
    %create Gram matrices (precomputing):
        for nkp = 1 : length(kp_v) %use this line without 'Matlab: parallel computing'
        %parfor nkp = 1 : length(kp_v) %use this line with 'Matlab: parallel computing'
            %compute G:
                kp = kp_v(nkp);
                co = K_initialization(cost_name,1,{'kernel',kernel,'sigma',kp});
                G = compute_Gram_matrix(X,co);
            %save G:
                FN = FN_Gram_matrix_v(dataset,cost_name,kernel,kp,base_kp,v);
                save_Gram_matrix(FN,G);
        end

    %output values (Y_train, Y_val, Y_test): Z: Classification values used
    % instead of Regression values Y
        Y_train = Z(idx_train);
        Y_val = Z(idx_val);
        Y_test = Z(idx_test);     
        nactive_rp = length(rp_v);
for v = v_v
    %validation surface, test surface:
        %initialization:
            validation_surface = zeros(length(rp_v),length(kp_v));
            test_surface = zeros(length(rp_v),length(kp_v));
        %for nkp = 1 : length(kp_v)
            %G_train:
                nkp=1;
                kp = kp_v(nkp);
                %load G => G_train:
                    % FN = FN_Gram_matrix_v(dataset,cost_name,kernel,kp, base_kp,v);
                    % only one kernel parameter is assumed.
                    FN = FN_Gram_matrix_v(dataset,cost_name,kernel,kp, base_kp,1);
                    load(FN,'G');
                    G_train = G(idx_train,idx_train); %Gram matrix of the training distributions  
            %validation_slice, test_slice:
                validation_slice = zeros(length(rp_v),1);
                test_slice = zeros(length(rp_v),1);
                for nrp = 1 : length(rp_v)    %use this line without 'Matlab: parallel computing'
                %parfor nrp = 1 : length(rp_v) %use this line with 'Matlab: parallel computing'
                    rp = rp_v(nrp);
                    %left:
                        A = real(inv(G_train + L_train * rp * eye(L_train))); %real(): to avoid complex values due to epsilon rounding errors
                        left = Y_train.' * A; %left hand side of the inner product; row vector
                    %Y_predicted_val, Y_predicted_test:
                        Y_predicted_val = sign( left * G(idx_train,idx_val) ).';  %for classification 
                        Y_predicted_test = sign( left * G(idx_train,idx_test) ).';%column vector
                        % Y_predicted_val = ( left * G(idx_train,idx_val) ).';  %column vector
                        % Y_predicted_test = ( left * G(idx_train,idx_test) ).';%column vector
                    %update the validation and test slices:
                        %L_2 error:
                            %validation_slice(nrp) = norm(Y_val-Y_predicted_val);
                            %test_slice(nrp) = norm(Y_test-Y_predicted_test);
                        %RMSE:
                            validation_slice(nrp) = RMSE(Y_val,Y_predicted_val);
                            test_slice(nrp) = RMSE(Y_test,Y_predicted_test);
                end
                validation_surface(:,nkp) = validation_slice;            
                test_surface(:,nkp) = test_slice;            
        %end

    %optimal parameters (regularization-, kernel parameter):
        [rp_idx,kp_idx,minA] = min_2D(validation_surface);
        kp_opt = kp_v(kp_idx);
        rp_opt = rp_v(rp_idx);    
        objective_opt = test_surface(rp_idx,kp_idx);
        objective_values(v) = objective_opt;
        active_rp = rp_v(nactive_rp);    % query using current rp
        
          
    %plot (validaton_surface, test_surface):
        plot_surf(validation_surface,kp_v,rp_v,base_kp,base_rp,'log(validation surface)');
        plot_surf(test_surface,kp_v,rp_v,base_kp,base_rp,'log(test surface)');
        figure;
            %X_parameter_test:    
                X_parameter_test = X_parameter(idx_test);            
            %Y_test_predicted:
                %load G => G_train:
                    %FN = FN_Gram_matrix_v(dataset,cost_name,kernel,kp_opt,base_kp,v);
                    % testing for only a single kp
                    FN = FN_Gram_matrix_v(dataset,cost_name,kernel,kp_opt,base_kp,1);
                    load(FN,'G');
                    G_train = G(idx_train,idx_train); %Gram matrix of the training distributions  
                %left:
                    A = real(inv(G_train + L_train * rp_opt * eye(L_train))); %real(): to avoid complex values due to epsilon rounding errors
                    left = Y_train.' * A; %left hand side of the inner product; row vector            
                Y_predicted_test = ( left * G(idx_train,idx_test) ).'; %column vector
            plot(X_parameter,Y,'b',X_parameter_test,Y_predicted_test,'*g'); 
            %decorations:
                %title:
                    kp_str = num2str(eval(strcat(['log',num2str(base_kp),'(',num2str(kp_opt),')'])));
                    rp_str = num2str(eval(strcat(['log',num2str(base_rp),'(',num2str(rp_opt),')'])));
                    title_str = strcat('Prediction with the opt. parameters: log_{',num2str(base_kp),'}(kernel par.)=',kp_str,', log_{',num2str(base_rp),'}(reg. par.)=',rp_str);
                    title(title_str);
                %labels, legend:
                    xlabel('Rotation angle (\alpha)');
                    ylabel('Entropy of the first coordinate');
                    legend({'true','predicted'});
    % query 
        [idx_query] = ...
              QUIRE(G,idx_train,idx_unlabel, Y_train, active_rp );
        [idx_train] = [idx_train,idx_query];
        L_train = L_train +1;
        Y_train = Z(idx_train);
        idx_unlabel = setdiff(idx_unlabel,idx_train);
    % update active_rp
        if nactive_rp > 1, nactive_rp = nactive_rp -1; end;
end

objective_values,

cd(dir_present);%change back to the original directory, from 'dir_G'
%-------------------------------
%matlabpool close; %use this line with 'Matlab: parallel computing'