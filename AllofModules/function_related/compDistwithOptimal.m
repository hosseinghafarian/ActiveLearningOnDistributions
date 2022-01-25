function [distX, distwo, distst, distalpha,diffXMat] = compDistwithOptimal(x_opt, alpha_opt, x_curr, alpha_curr)
    [Xapproxcurr,pcurr,qcurr,qyucurr,wocurr,stcurr]     = getParts(x_curr);
    [Xapproxopt,popt,qopt,qyuopt,woopt,stopt]           = getParts(x_opt);
    distX                 = norm(Xapproxcurr-Xapproxopt)/norm(Xapproxopt);
    distwo                = norm(wocurr-woopt)/norm(woopt);
    distst                = norm(stcurr-stopt)/norm(stcurr); % this value is not correct since we always assign stopt=0 
    distalpha             = norm(alpha_opt-alpha_curr)/norm(alpha_opt);
    
    diffXMat              = Xapproxcurr-Xapproxopt;
end