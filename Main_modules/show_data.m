function  show_data(profile, data, initLexp)
    if profile.showOptions.showData
        xTrain = data.X;
        yTrain = data.Y;
        showOptions.showContour = true;
        %showOptions.model       = ModelAndData.model;
        showOptions.showIndex   = true;
        showOptions.isUpdate    = true;
        sampleShow(showOptions.figName, xTrain, yTrain', initLexp, profile.showOptions);
    end
end