function [data]      = load_dataobject(datasetId, profile, recompute)
if nargin == 2
    recompute = false;
end
if ~profile.distributional
    % Data set Load
    [data]              = loaddataset(datasetId, profile, recompute);

    data.isDistData     = false; % data is not distributional
    %         [data          ]    = LB_mapping_func(data, LB_settings);
    [data]              = data.DS_subsampling_func(data, data.DS_settings);
    assert(numel(data.Y)==numel(data.F),'Error index of indices and the number of indices doesnot match');
    
    %     end
else
    [data            ]  = loadDISTdataset(datasetId, profile, recompute); 
    data.isDistData     = true;
end
if ~isfield(data, 'isTwoLevel')
    data.isTwoLevel = false;
end
if ~isfield(data, 'data_comp_kernel')
    data.data_comp_kernel = @comp_simple_kernel_general;
end    
checkdata(data);
end