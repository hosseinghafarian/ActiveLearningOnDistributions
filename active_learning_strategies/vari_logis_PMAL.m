function [query_id] = vari_logis_PMAL(trainsamples, initL, samples_to_query_from, K, lambda, learningparams, alparams) 
global cnstData
   %xtrain = trainsamples.X;
   ytrain = trainsamples.Y; 
   alphavar = alparams{1};
   
   n = size(K, 1);
   
   
   [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);
   y_Lindex   = ytrain(Lindex);
   assert(n==numel(Lindex), 'wrong number of instances in vari_logic_PMAL');
   
   if isfield(learningparams, 'prexi')
       prexi = learningparams.prexi;
   else
       prexi = zeros(n, 1);
   end
   
   [queryind] = vlog_PMAL( Lind, Uind, y_Lindex, alphavar );
   
   %query_id = trainsamples.F_id(queryind);
   query_id                                   = get_ID_fromind(trainsamples, queryind);
   assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in the distributions ID in learning data');


   function [queryind] = vlog_PMAL(Lind, Uind, y_Lindex, alphavar )


      n_u = numel(Uind);
      mlogpyx_yq_yu = zeros(n_u, 1);
      for q = 1:n_u
          [mlogpyx_yq_yu(q), prexi] = comp_maxyq(K, Lind, q, Uind, y_Lindex, alphavar, prexi);
      end
      [~, indq] = min(mlogpyx_yq_yu);
      queryind = Uind(indq);
   end
    function [val, prexi] = comp_maxyq(K, Lind, q, Uind, y_Lindex, alphavar, prexi)
        qind          = Uind(q);
        Lindq         = [Lind, qind];
        Uindmiq       = [Uind(1:q-1), Uind(q+1:end)];
        
        y_q        = 1;
        y_Lindexplusq = [y_Lindex, y_q];
        [val1 , xi1] = comp_maxyq_minyu(K, Lindq, Uindmiq, y_Lindexplusq, alphavar, prexi);
        
        y_q        = -1;
        y_Lindexplusq = [y_Lindex, y_q];
        [valm1 , xim1] = comp_maxyq_minyu(K, Lindq, Uindmiq, y_Lindexplusq, alphavar, prexi);
        
        if val1> valm1
            val = val1; prexi = xi1;
        else
            val = valm1; prexi = xim1;
        end
    end
    function [val , xi] = comp_maxyq_minyu(K, Lindq, Uindmiq, y_Lq, alphavar, prexi)
       tol = 0.001;
       conv = false;
       y = zeros(size(K,1), 1);
       while ~conv
           Mxi = computeL(K, prexi, alphavar);
           %y_U = inv(L(Uindmiq, Uindmiq))*L(Uindmiq, Lindq)*y_Lq;
           y_U = Mxi(Uindmiq, Uindmiq)\(Mxi(Uindmiq, Lindq)*y_Lq');

           y(Uindmiq) = y_U;
           y(Lindq)   = y_Lq;
           mu  = 0.5* Mxi* y;
           
           xi  = sqrt(diag(max(Mxi, 0))+ mu.*mu);
           relnorm = norm(xi-prexi)/(norm(xi)+0.00000001);
           if  relnorm <tol
              conv  = true; 
           end    
           
           prexi = xi;

       end
       %%%%Warning: the following objective function doesn't consider
       %%%%ln\vert S_n\vert in the objective function. It may be
       %%%%inaccurate.
       diagxi = 2 *diag(lambdaofxi(xi));
       detdif = log(det(diagxi))- log(det(diagxi+ 1/alphavar* K));
       val =  1/8*y'*Mxi*y + sum( log(logisticfunc(xi)) - 0.5* xi + 0.5*(logisticfunc(xi)-0.5).*xi) + 1/2* detdif;     
    end
    function Lxi = computeL(K, xi, alphavar)
       eps = 0.0000001;
       
       DalphaK = alphavar*K+diag(oneoverlamdbaxi(xi))+eps* eye(n);
       
       Lxi = alphavar*( K- alphavar* (K/DalphaK)*K)  +eps* eye(n);
    end
    function [ olambdaxi ] = oneoverlamdbaxi(xi)
        denom = feval(@logisticfunc, xi)-0.5*ones(numel(xi),1);
        olambdaxi = xi./denom;
        zeroinds  = abs(xi)<0.0000000001;
        olambdaxi(zeroinds) = 0; 
    end
end
function [lamxi] = lambdaofxi(xi)
tol = 0.0000000001;
     lamxi = feval(@logisticfunc, xi)- 0.5*ones(numel(xi),1);
     xinz  = xi + tol * ones(numel(xi),1);
     lamxi = lamxi./xinz;
end
function [val] = logisticfunc( x )
 val = 1./(1+exp(-x));
end