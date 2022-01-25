function [convergemeasure_row, dist_row, f_row] = progress_report(learningparams,iter,moditer, verbose, x_hatbar, x_hatbar_pre, dualvars, dualvars_pre, alpha_hatbar, alpha_hatbar_pre, x_k, alpha_k)
global optimals
        convergemeasure_row(1,1) = norm(alpha_hatbar-alpha_hatbar_pre)/(1+norm(alpha_hatbar_pre));
        convergemeasure_row(1,3) = euclidean_dist_of_duals(dualvars, dualvars_pre);
        convergemeasure_row(1,2) = euclidean_dist_of_x(x_hatbar, x_hatbar_pre);
        convergemeasure_row(1,4) = convergemeasure_row(1,1) + convergemeasure_row(1,2) + convergemeasure_row(1,3);         
        [f_row(1,1)]             = f_of_xAndAlpha(x_hatbar,alpha_hatbar,learningparams);
        [f_row(1,2)]             = f_of_xAndAlpha(x_k,alpha_k,learningparams);
        [dist_row(1,1), dist_row(1,2), dist_row(1,3), dist_row(1,4), diffXMat] = compDistwithOptimal(optimals.x_opt, optimals.alpha_opt,  x_hatbar,alpha_hatbar);
        
        if (mod(iter,moditer)==1)&&verbose
strtitle = sprintf('iter | conv  | x conv   |dualconv |alphaconv| x.u-Opt|w_o-Opt |x.st-Opt|alpha-Opt|f_hatbar| f_k  |f_opt ');
             disp(strtitle);
        end
        if verbose
            str = sprintf('%4.0d |%7.4f|%7.4f   |  %7.4f|  %7.4f| %7.4f| %7.4f| %7.4f|  %7.4f|%7.4f |%3.0d|%4.3f|%9.6f',...
                           iter, convergemeasure_row(1,4), convergemeasure_row(1,2), ...
                                 convergemeasure_row(1,3), convergemeasure_row(1,1), ...
                                 dist_row(1,1), dist_row(1,2), dist_row(1,3), dist_row(1,4),...
                                 f_row(1,1),f_row(1,2),optimals.obj_opt);
            disp(str);
        end
end