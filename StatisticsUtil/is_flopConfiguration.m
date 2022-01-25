function [isflop] = is_flopConfiguration(T_s, s, S, beta_l, alpha_l)
pi0 = 0.5;
pi1 = 0.5*nthroot((1-beta_l)/alpha_l, S);
logpi = log((1-pi1)/(1-pi0));
logpi10 = log(pi1/pi0);
denom   = logpi10-logpi;
a   = log(beta_l/(1-alpha_l))/denom;
b   = log((1-pi0)/(1-pi1))/denom;

isflop = (sum(T_s) <= a + b*s);
end