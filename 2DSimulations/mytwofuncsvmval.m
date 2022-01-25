function [ret_ypred] = mytwofuncsvmval(Sigma,xtest,xapp,yapp,info,lambda,lambda_o,wlambda)
switch info.type
    case {1,2,3}    
        ONESn=ones(size(xapp,1),1);
        K=kernel(xapp,xtest,Sigma);
        if info.primal 
            ypred = sum((info.primalWalpha)'*K,1);
            %ypredf = ypredwsum((info.primalfalpha)'*K,1);
            ypred_o= sum((info.primalW_ohr)'*K,1);    
        else
            alpha_t=info.alpha_r;
            ypred  = 1/(2*lambda)*sum((alpha_t.*yapp)'*K,1);
            ypred_o=ypred;
            switch info.type
                case 1
                    ypred_o = 1/(2*lambda_o)*sum((alpha_t.*yapp+info.h-info.r)'*K,1);
                case 3
                    ypred_o = 1/(2*lambda_o)*sum((info.alphaeta.*yapp+info.h-info.r)'*K,1);
            end
        end
        ret_ypred.type    = info.type;
        ret_ypred.ypred   = ypred;
        ret_ypred.ypred_o = ypred_o;
        % ypred_o = 1/(2*lambda_o)*sum(alpha_t.*yapp*kernel(xapp,xtest),1);
    case 4
        ONESn=ones(size(xapp,1),1);
        K=kernel(xapp,xtest,Sigma);
        if info.primal==true 
            ypredw = sum((info.primalWalpha)'*K,1);
            ypredf = sum((info.primalfalpha)'*K,1);
            ypred_o= sum((info.primalW_ohr)'*K,1);    
        end
        if info.dual==true
            ypredw = 1/(2*lambda)*sum((info.beta_r.*yapp+info.mu_r-info.nu_r)'*K,1);
            ypredf = 1/(2*lambda)*sum((info.alpha_r.*yapp+info.eta_r+info.nu_r-info.mu_r-info.s)'*K,1);
            ypred_o= 1/(2*lambda_o)*sum((info.alpha_r.*yapp+info.r-info.h)'*K,1);       
        end        
        ret_ypred.type    = info.type;
        ret_ypred.ypred   = ypredw;
        ret_ypred.ypredf  = ypredf;
        ret_ypred.ypred_o = ypred_o;
    case 5
        ONESn=ones(size(xapp,1),1);
        K=kernel(xapp,xtest,Sigma);
        if info.primal==true 
            ypredw = sum((info.primalWalpha)'*K,1);
            
            ypred_o= sum((info.primalW_ohr)'*K,1);    
        end
               
        ret_ypred.type    = info.type;
        ret_ypred.ypred   = ypredw;
        %ret_ypred.ypredf  = ypredf;
        ret_ypred.ypred_o = ypred_o;
    case 7
        ONESn=ones(size(xapp,1),1);
        K=kernel(xapp,xtest,Sigma);
        g=1-yapp.*info.primalw_oxT_i';
        if info.primal==true 
            ypredw = sum((yapp.*info.primalWalpha.*g)'*K,1);
            
            ypred_o= sum((info.primalW_ohr)'*K,1);    
        end
               
        ret_ypred.type    = info.type;
        ret_ypred.ypred   = ypredw;
        %ret_ypred.ypredf  = ypredf;
        ret_ypred.ypred_o = ypred_o;
    case 8
        ONESn=ones(size(xapp,1),1);
        K=kernel(xapp,xtest,Sigma);
        g=1-yapp.*info.primalw_oxT_i';
        if info.primal==true 
            ypredw = sum((yapp.*info.primalWalpha.*g)'*K,1);
            
            ypred_o= sum((info.primalW_ohr)'*K,1);    
        end
               
        ret_ypred.type    = info.type;
        ret_ypred.ypred   = ypredw;
        %ret_ypred.ypredf  = ypredf;
        ret_ypred.ypred_o = ypred_o;        
        
    case 9
        ONESn=ones(size(xapp,1),1);
        K=kernel(xapp,xtest,Sigma);
        g=1-yapp.*info.primalw_oxT_i';
        if info.primal==true 
            ypredw = sum((yapp.*info.primalWalpha.*g)'*K,1);        
            ypred_o= sum((info.primalW_ohr)'*K,1);    
        end
               
        ret_ypred.type    = info.type;
        ret_ypred.ypred   = ypredw;
        %ret_ypred.ypredf  = ypredf;
        ret_ypred.ypred_o = ypred_o;
end
end        
function Ker=kernel(x1,x2,Sigma)
    Ker= exp(- pdist2(x1,x2)/(2*Sigma^2));
end