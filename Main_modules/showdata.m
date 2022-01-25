function fig = showdata(data, showOptions, fig, func, model, methodname, learningparams)
fig = [];
if data.isDistData 
    return
end
if showOptions.showData
    if nargin == 2
       [fig,~,~] = sampleShow([], showOptions.figName, data.X, data.Y, [], showOptions);
    else
       showOptions.showContour = true;
       [fig,~,~] = sampleShow(fig, showOptions.figName, data.X, data.Y, [], showOptions,...
                              func, model, methodname, learningparams);
    end
end
end