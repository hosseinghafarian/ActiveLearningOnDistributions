function [h]=ShowTransformSpace(h,X,Z,TS,initL,Sf,showOptions,isUpdate)
if nargin== 4
    n=size(X,2);
    initL=[1:n];
    Sf=Z;
    showOptions.clear=true;
    showOptions.acceptNewInstance=true;
    showOptions.selectInstance = true;
    showOptions.StopforKey     = false;
    showOptions.isUpdate = false;
else
    showOptions.isUpdate = isUpdate;
end
figName= 'Transform Space';
[fig,outp,selectedsam]=sampleShow([],figName, TS', Z ,initL,showOptions);

% figure(h);
% 
% if nargin <4
%     h=scatter(TS(:,1),TS(:,2),100,Z,'fill');    
% else 
%     c=Z-Sf;c=c/2;
%     d=logical(c);
%     Z(d)=2;
%     h=scatter(TS(:,1),TS(:,2),100,Z,'fill');
% end
 
end