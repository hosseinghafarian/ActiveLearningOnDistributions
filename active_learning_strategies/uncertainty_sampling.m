% UNCERTAINTY_SAMPLING queries the most uncertain point.
%
% This is an implementation of uncertainty sampling, a simple and
% popular query strategy. Uncertainty sampling successively queries
% the point with the highest marginal entropy:
%
%   x* = argmax H[y | x, D],
%
% where H[y | x, D] is the entropy of the marginal label
% distribution  p(y | x, D):
%
%   H[y | x, D] = -\sum_i p(y = i | x, D) \log(p(y = i | x, D)).
%
% Usage:
%
%   query_ind = uncertainty_sampling(problem, train_ind, observed_labels, ...
%           test_ind, model)
%
% Inputs:
%
%           problem: a struct describing the problem, containing fields:
%
%                  points: an (n x d) data matrix for the available points
%             num_classes: the number of classes
%
%         train_ind: a list of indices into problem.points indicating
%                    the thus-far observed points
%   observed_labels: a list of labels corresponding to the
%                    observations in train_ind
%          test_ind: a list of indices into problem.points indicating
%                    the points eligible for observation
%             model: a function handle to a probability model
%
% Output:
%
%   query_ind: an index into test_ind indicating the point to query
%              next
function query_ind =uncertainty_sampling(xTrain, yTrain, Lind, Uind)

  classes  =[-1 ;1 ];
  s= 0.1;
  k = 7;
  
  probabilities = knn_model(classes,xTrain, yTrain,Lind,k,s);
  
  scores = marginal_entropy(probabilities);

  [~,ind] = max(scores(Uind));
  query_ind = Uind(ind);
end
function scores = marginal_entropy( probabilities)

  % remove any zeros from probabilities to approximate 0 * -inf = 0
  probabilities = max(probabilities, 1e-100);

  scores = -sum(probabilities .* log(probabilities), 2);

end
function probabilities = knn_model(classes,xTrain, yTrain,initL,k,s)

  num_test = numel(yTrain);
  num_classes  = numel(classes);
  probabilities = zeros(num_test, num_classes);
  n = numel(yTrain);
  %% assume we have kernel function. 
  
  IDX = knnsearch(xTrain',xTrain','K',k);
  test_ind = 1:n;
  % accumulate weighted number of successes for each class
  for i = 1:num_classes
    class_c = yTrain(initL) ==classes(i) ;
    initLclass = initL(class_c);
    isl = zeros(numel(test_ind),1);
    t =1;
    for j=test_ind
        testidx= IDX(j,:);
        try
           %isl(t) = sum(ismember(initLclass,testidx));
           isl(t) = ~isempty(intersect(initLclass', testidx)); 
        catch
           warning('Something wrong');
        end
        t=t+1;
    end
    probabilities(:, i) = s;
    
    probabilities(test_ind, i) = probabilities(test_ind, i)+ sum(isl,2);
  end

  % normalize probabilities
  probabilities = bsxfun(@times, probabilities, 1 ./ sum(probabilities, 2));

end