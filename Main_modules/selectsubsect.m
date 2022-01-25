function [ sellogic ] = selectsubsect( Y, selectlabel , max_size)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    n         = numel(Y);
    poslogic  = (Y==selectlabel);
    posind    = find(poslogic);
    n_pos     = sum(poslogic);
    n_sp      = min(n_pos, max_size);
    selpos    = randperm(n_pos, n_sp);
    selind    = posind(selpos);
    sellogic  = false(n,1);
    sellogic(selind) = true;
end

