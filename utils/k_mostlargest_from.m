function [Ind, success] =k_mostlargest_from(Array ,k, from)
    k = floor(k);
    
    if k==0,  Ind = []; return; end
    Ind       = zeros(k,1);
    Arraycopy = Array;
    success   = true;
    m_inf_ind = 1:numel(Array);
    m_inf_ind = setdiff(m_inf_ind, from);
    Arraycopy(m_inf_ind) = -inf;
    for j= 1:k
       [a, Ind_t] = max(Arraycopy);
       if a>-inf
           Ind(j) = Ind_t;
       else
           success = false;
           return
       end
       Arraycopy(Ind_t) = -inf;
    end
    maximumValues = Array(Ind);
end