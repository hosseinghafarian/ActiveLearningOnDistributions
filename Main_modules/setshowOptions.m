function showOptions                   = setshowOptions()
    %lOptions.threshold = 14;
    %lOptions.lowBValue = -1;
    %lOptions.highBValue= 1;
    % showOptions
    showOptions.showData   = false ;    % To show data in 2d experiments or not?
    showOptions.viewTransformSpace = true;
    showOptions.clear=true;
    showOptions.acceptNewInstance=false;
    showOptions.selectInstance = false;
    showOptions.StopforKey     = false;
    showOptions.isUpdate = false;
    showOptions.showContour = false;
    showOptions.model= 0;
    showOptions.showIndex   = false;
    showOptions.figName='Samples';
end