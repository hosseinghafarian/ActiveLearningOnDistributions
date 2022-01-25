function [Z,v]                         = projon_Conestar(query, R, g,s_IC,s_IV, y_IC,y_IV,nSDP,n_S)
    % These two projections are based on the moreau decomposition
    % only project some elements of the matrix and not any of the appended
    % vector
    % indices of non-negativity constraints encoded in Z: non negativity
    % constraints is enforced using Z
    [ind] = ind_of_nonnegativityConstraints(query, nSDP);
    R_project = max(R(ind),0);
    Z         = zeros(size(R));
    Z(ind)    = R_project - R(ind);
    y_I       = [y_IC;y_IV];
    s_I       = [s_IC;s_IV];
    vp        = min(g-y_I,s_I); %\mathcal{K}is \forall s\in K, s<=s_I
    v         = (g-y_I)- vp;  
end
