function [ first_lab, sec_lab, max_ind ] = one_vs_one( stlabel, enlabel )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
k = 1;
for lab1 = stlabel:enlabel-1
    for lab2 = lab1+1:enlabel
       first_lab(k) = lab1;
       sec_lab(k)   = lab2;
       k            = k + 1;
    end
end
max_ind = k -1;
end

