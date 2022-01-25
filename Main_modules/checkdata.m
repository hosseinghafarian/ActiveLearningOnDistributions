function checkdata(data)
    iscorrect    = (sum(~ismember(data.F, data.F_id))==0)&(sum(~ismember(data.F_id, data.F))==0);
    assert(iscorrect, 'Error, dist_id is not in data.F_id');
    F_idsuniqe   = (numel(data.F_id) == numel(unique(data.F_id)));
    assert(F_idsuniqe, 'Error, data of F_id isnot unique');
    samesize     = (numel(data.F_id)==data.n);
    if ~isfield(data, 'isMultiTask')
        data.isMultiTask = false;
    end
    if data.isMultiTask 
        samesize = samesize &(numel(data.Y)==numel(data.F));
    end
    haveYlabel = isfield(data,'Y')|(data.isMultiTask); 
    assert(haveYlabel, 'data must have a field Y as label or must be multitask which is not.');
    if isfield(data,'Y')&& ~data.isMultiTask
        samesize = samesize &(numel(data.Y)==data.n);
        assert(size(data.Y, 1)==1,'Y must be a row vector of labels');
    end
    if isfield(data,'noisy')&& isfield(data,'labelnoise')&&~data.isMultiTask
        samesize     = samesize &(numel(data.noisy)==data.n)&...
                            (numel(data.labelnoise)==data.n);
    else
        samesize     = samesize &(numel(data.noisy)==numel(data.F))&...
                            (numel(data.labelnoise)==numel(data.F));
    end                    
    assert(samesize, 'Error, size of data.F_id and data.n are not equal');
  
    samelength   = (numel(data.F)==size(data.X,2));
    assert(samelength, 'Error, size of data.F and data.X doesnot match');
    
    assert(isfield(data,'C')&&isfield(data,'binary'),'C and binary fields does not exist');
    if data.binary && isfield(data,'Y')
        assert(numel(unique(data.Y))==2, 'data is not binary while the flag data.binary shows such');
        nclassBinary = (numel(unique(data.Y))==2)&data.binary;
        %assert(~nclassBinary, 'Cannot have more than two class Y and perform binary classification');
    end
    if isfield(data, 'isTwoLevel') && data.isTwoLevel
        assert(isfield(data, 'Y_L2')&&isfield(data, 'L2labelflag'), 'Y_L2 or L2lableflag is not exist');
        assert(numel(data.Y_L2)==numel(data.F), 'labels of level 2 are not the same size as level one, use L2labelflag when there is no label for some examples in level2');
        assert(numel(data.L2labelflag)==numel(data.F), 'L2lable flag is not the same size as F');
    end
end