function [measure] = comp_measure(model, acc, measure, final, number, comp_acc)
fieldslist = {'noise_detect_acc', 'maxnoise_p', 'avg_noise_p', 'avg_nonnoise_p'};
if nargin== 3 
   if isempty(measure) 
       measure.acc            = acc; 
       
       
       fl  = isfield(model, fieldslist);
       for i = 1:numel(fl)
           if fl(i)
              measure.(fieldslist{i}) = model.(fieldslist{i}); 
           end
       end
   else
       measure.acc            = [measure.acc,acc]; 
       fl  = isfield(model, fieldslist);
       for i = 1:numel(fl)
           if fl(i)
              measure.(fieldslist{i}) = [measure.(fieldslist{i}),model.(fieldslist{i})]; 
           end
       end
   end
elseif nargin==6
   if isempty(measure) 
       measure.acc            = acc; 
       measure.cmp_acc        = comp_acc;
       fl  = isfield(model, fieldslist);
       for i = 1:numel(fl)
           if fl(i)
              measure.(fieldslist{i}) = model.(fieldslist{i}); 
           end
       end
   else
       measure.acc            = [measure.acc,     acc     ]; 
       measure.cmp_acc        = [measure.cmp_acc, comp_acc]; 
       fl  = isfield(model, fieldslist);
       for i = 1:numel(fl)
           if fl(i)
              measure.(fieldslist{i}) = [measure.(fieldslist{i}),model.(fieldslist{i})]; 
           end
       end
   end   
elseif final && number > 0
   [measure.acc_avg, measure.acc_std] = stat(measure.acc);
   if isfield(measure, 'cmp_acc')
      [measure.cmp_acc_avg, measure.cmp_acc_std] = stat(measure.cmp_acc);
   end
   fl  = isfield(model, fieldslist);
   for i = 1:numel(fl)
       if fl(i)
          [measure.([fieldslist{i},'_avg']), measure.([fieldslist{i},'_std'])] = stat(measure.(fieldslist{i})); 
          measure.(fieldslist{i}) = measure.(fieldslist{i});
       end
   end
elseif final && number == 0 
   measure.acc_avg = -Inf;
   measure.acc_std = -Inf;
       
end
end
