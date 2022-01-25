function [queryind, model_set, Y_set] = get_query_label_of_x(x_k)
%% Discussion about using use of qyu for obtaining q
            %y_ui may be less than 1 or greater than -1. In these cases,
            %q value may mislead us. 
            % I think the best value is this value which is the nearest
            % value to y_ui*(1-q_i): may be that's because r is not exactly
            % the absolute value of V(n_l+1:n,nap+1) and y_ui is relaxed to
            % [-1,1],but we must assume it is {-1,1}
            
            %ra must be equal to absoulte value of qyu but in pratice the
            %value is completely different due to relaxation of y_ui to [-1,1] 
            %sa    =value(s);
%             if formp==1
%                  qresult=1-abs(qyu);
%                  qresult=qresult.*(1-pv(n_l+1:n));
%             else
%                  qresult=qv;    
%             end
global cnstData
% Obtaining results
    n         = cnstData.n_S;
    nSDP      = cnstData.nSDP;
    [qv, qyu] = get_q_g_of_x(x_k);
    pv        = get_p_of_x(x_k);

    % find largest p, (they are the noisiest)
    separate          = true;
    [noisysamplesInd] = get_noisy_sample_ind(pv, n, cnstData.lnoiseper, cnstData.onoiseper, separate);
    % retain only unlabeled samples
    isunlabelednoisy  = ismember(noisysamplesInd, cnstData.unlabeled);
    noisysamplesInd   = noisysamplesInd(isunlabelednoisy);
    alsubtype         = 4; % this is used so far.
    alsubtype         = 1; % just for test
    query_subset      = cnstData.query;
    qyu_subset        = qyu(query_subset);
    pv_subset         = pv(query_subset);
    qresult           = zeros(n,1);
    if alsubtype==1
        qresult(query_subset) = qv;
    elseif alsubtype==2
        qresult(query_subset) = qv;
        qresult = qresult.*(1-pv);
    elseif alsubtype==3
        qresult(query_subset) = 1-abs(qyu_subset);
        qresult = qresult.*(1-pv);                
    elseif alsubtype==4
        qresult(query_subset) = 1-abs(qyu_subset);
        qresult = qresult.*(1-pv);
        qresult(noisysamplesInd) = 0;          % TODO: it must be checked that p_i for noisydata are significantly larger than others,
                                               % otherwise if for example all of them are zero it has no meaning
    elseif alsubtype==5
        qresult = qv.*(1-pv);
        qresult(noisysamplesInd) = 0; 
    elseif alsubtype==6
        qresult(query_subset) = qv;    
        qresult(noisysamplesInd) = 0;
    end
    tq       = k_mostlargest(qresult, cnstData.batchSize);
    queryind = tq;%unlabeled(tq); 
    Y_set    = sign(value(qyu)); 
    model_set.G                = reshape(x_k.u(1:nSDP*nSDP),nSDP,nSDP);
    h                = 1-pv;
    h(query_subset)  = h(query_subset)-qv;
    model_set.h      = h;
    model_set.noisySamplesInd = noisysamplesInd;
end