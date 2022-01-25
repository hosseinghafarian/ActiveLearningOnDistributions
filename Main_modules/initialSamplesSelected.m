function [initL ]=initialSamplesSelected(Options, initType, TrainSamples)
%% Initial Samples Selection
    % How to initialy select random samples? 
    % 1: Select Two random samples
    % 2: Select Two random samples from different classes
    % 3: All of samples (Passiver learning)
    % 4: select sample by user. only when data is 2d and Synthesised
    % 5: select n_initial_samples/2 from non noisy instances. 
if ~TrainSamples.isDistData
    if  initType==1 % select sample no 1 
        Ts = size(TrainSamples.X,2);
        f  = randi(Ts);
        s  = randi(Ts);
        initL = [f,s]';
        return;
    end
    if  initType==2 % select two samples from different classes
        Ts=size(TrainSamples.X,2);
        f=randi(Ts);
        s=randi(Ts);
        % if two points selected randomly have the same class, search for 
        % another one with different class
        count =1;
        while (TrainSamples.Y(s)==TrainSamples.Y(f))
            s=randi(Ts);
            count=count+1;
            if count > 20, 
                disp('error cannot find two instances from different classes');
                initL = [s];
                return;
            end
        end
        initL=[f,s]';
        return
    end
    if  initType==5 % select Options.n_initial_samples samples from different classes assuming two class only. 
       Ts=size(TrainSamples.X,2); 
       [initLind] = get_initial(Ts);
       initL   = sort(TrainSamples.F_id(initLind));
       return
    end
    if  initType==3 % use all of samples in the first place
        initL=[1:size(xTrain,2)]';
    end
    if  initType==4 % user select samples
        shwSOp.clear=false;
        shwSOp.acceptNewInstance=false;
        shwSOp.selectInstance = true;
        shwSOp.StopforKey     = true;
        shwSOp.isUpdate = false;
        shwSOp.onSpecificFigure=false;
        shwSOp.showContour = false;
        shwSOp.showIndex   = false;
        [outp,initL]=sampleShow('Selecting Samples initially labeled for Active Learning',TrainSamples.X, TrainSamples.Y,[],shwSOp);
    end
else
   initLind= get_initial(TrainSamples.n);
   initL   = sort(TrainSamples.F_id(initLind));
   return    
end
function [initL] = get_initial(n)
       n_initial_select = Options.n_initial_samples;
       n_each_class     = floor(n_initial_select/2);
       %f=randi(Ts);
       f_list = zeros(n_each_class,1);
       s_list = zeros(n_each_class,1);
       %f_list(1) = f;
       n_f    = 0;
       n_s    = 0;
       % if two points selected randomly have the same class, search for 
       % another one with different class       
       itemlist = randperm(n);
       i        = 1; 
       while ((n_f < n_each_class || n_s < n_each_class)&& i<=n)
           new_item = itemlist(i);
           if ~TrainSamples.noisy(new_item)
               if n_f== 0 
                   f             = new_item;
                   f_list(1)     = f;
                   n_f           = n_f + 1;
               elseif TrainSamples.Y(new_item)==TrainSamples.Y(f) && n_f < n_each_class && ~ismember(new_item, f_list)
                   f_list(n_f+1) = new_item;
                   n_f           = n_f + 1;
               elseif TrainSamples.Y(new_item)~=TrainSamples.Y(f) && n_s < n_each_class && ~ismember(new_item, s_list)
                   s_list(n_s+1) = new_item;
                   n_s           = n_s + 1;
               end
           end
           i  = i + 1;
       end
       assert(n_s >= n_each_class && n_f >= n_each_class, 'error cannot find enough instances from each class');
%        if n_s < n_each_class || n_f < n_each_class 
%                 disp('error cannot find enough instances from each class');
%                 initL = [f_list(f_list>0); s_list(s_list>0)]';
%                 return
%        end
       initL=[f_list;s_list]';
end
end