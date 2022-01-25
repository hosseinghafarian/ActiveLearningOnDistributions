function c_alpha      = c_of_alpha(alpha_k, learningparams)
global cnstData
nSDP = cnstData.nSDP;
n_S  = cnstData.n_S;
    l_star_alpha = [alpha_k(1:n_S);zeros(nSDP-1-n_S,1)];
    eta_star     = learningparams.ca*[ones(n_S,1);zeros(nSDP-1-n_S,1)];
    G_starterm   = [-1/(2*learningparams.lambda)*(alpha_k*alpha_k'.*cnstData.KE),zeros(nSDP-1,1); ...
                    zeros(1,nSDP)];
    CALPHAMat    = diag([l_star_alpha+eta_star;0]) + G_starterm;
    c_alpha.u    = [reshape(CALPHAMat, nSDP*nSDP,1);zeros(n_S,1)];
    c_alpha.w_obeta = zeros(n_S,1);
    c_alpha.st   = 0;
end