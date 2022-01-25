function display_status(func, stid, str_in, lp, acc, durtime)
switch(stid)
    case 1  
       reportdisp = ['dataset,', str_in,' loaded'];
    case 2 
       reportdisp = str_in;
    case 3 
       reportdisp = ['searching for parameters for method:',str_in];
    case 4 
       datasetname = acc;
       str        = get_learning_string(lp);
       reportdisp = ['Doing kfold param:',str,' on dataset ', datasetname];
    case 5 
       foldi = num2str(str_in);
       kfold = num2str(lp);
       accstr= num2str(acc);
       reportdisp = ['fold ',foldi,' of ', kfold,' accuracy:',accstr];
    case 6
       foldi = num2str(str_in);
       kfold = num2str(lp);
       accstr= num2str(acc);
       durstr = num2str(durtime);
       reportdisp = ['fold ',foldi,' of ', kfold,' accuracy:',accstr, 'duration:', durstr];
end
disp([reportdisp,' ','in function ', func]); 
end