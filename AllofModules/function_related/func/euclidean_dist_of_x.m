function [squares] = euclidean_dist_of_x(x_k, Ghat)
global cnstData
squares = norm(x_k.u-Ghat.u)^2;
squares = squares + norm(x_k.st-Ghat.st)^2;
squares = squares + (x_k.w_obeta-Ghat.w_obeta)'*cnstData.Q*(x_k.w_obeta-Ghat.w_obeta);
end