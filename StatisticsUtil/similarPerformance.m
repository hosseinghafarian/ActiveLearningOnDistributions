function [issimilar] = similarPerformance(T_s, alphafast)
  [h,p,stats] = cochranqtest(T_s', alphafast);
  issimilar = (p <= alphafast);
end