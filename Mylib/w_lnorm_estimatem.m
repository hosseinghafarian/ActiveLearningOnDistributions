function  normwl_t   = w_lnorm_estimatem(KB_ll, KB_uu , KB_lu_z_l, ...
                                         R, F_A, F_id, mu_R_alli, R_reginv_alli, ...
                                         initL, z_l)
z_lKB_z_l = z_l'*KB_ll*z_l;
tic;
A_inv  = inv(KB_uu);
midx_A = numel(F_id);
normwl_t = zeros(midx_A,1);
for t=1:midx_A
    dist_id = F_id(t);
    ind_t   = (F_A== dist_id);
    n_P_t   = size(R_reginv_alli{t}, 1); 
    nu_t    = comp_nu_t(R, mu_R_alli, ind_t, F_A, F_id, midx_A, n_P_t, z_l, initL, dist_id);
    M_t     = R_reginv_alli{t};
    P_u_t   = get_P_u_t(R, mu_R_alli, ind_t, F_A, F_id, midx_A, n_P_t, z_l, initL);
    mu_t    = P_u_t'*M_t*nu_t;
    mu_t    = KB_lu_z_l- mu_t;
    %B_t     = P_u_t'*M_t*P_u_t;
    %mu_k    = it_psd(A_inv, B_t, mu_t);
    mu_k    = (KB_uu-P_u_t'*M_t*P_u_t)\mu_t;
    normwl_t(t) = z_lKB_z_l - nu_t'* M_t * nu_t-mu_t'*mu_k;
end
toc
end
%exactM = inv(cnstData.BayesK(cnstData.Uindex,cnstData.Uindex)-P_u_t'*M_t*P_u_t);
%   exactn(t)   = z_lKB_z_l - nu_t'* M_t * nu_t-mu_t'*exactM*mu_t;
%     [norm_mu_t, norm_mu_t1, aprM1, norm_mu_t2, aprM2]   = get_approx_byinv(n_P_t, mu_t, M_t, H, H_inv, KB_uu_inv, P_u_t);
%     norm(H'*exactM*H-aprM1,'fro')/norm(H'*exactM*H,'fro')
%     norm(exactM-aprM2,'fro')/norm(exactM,'fro')
function [norm_mu, norm_mu_t1, aprM1, norm_mu_t2, aprM2]  = get_approx_byinv(n_P_t, mu_t, M_t, H, H_inv, KB_uu_inv, P_u_t)
thrsholdsize = 30;
[norm_mu_t1, aprM1] = get_approx_by_eigs(mu_t, M_t, H, H_inv, KB_uu_inv, P_u_t);
[norm_mu_t2, aprM2] = get_approx_by_mul(mu_t, M_t, KB_uu_inv, P_u_t);
if n_P_t > thrsholdsize
    
    norm_mu = norm_mu_t1;
else
    
    norm_mu = norm_mu_t2;
end
end
function [norm_mu_t, aprx_Inv] = get_approx_by_eigs(mu_t, M_t, H, H_inv, KB_uu_inv, P_u_t)
%[V, D]  = eigs(KB_uu_inv*P_u_t'*M_t*P_u_t, 20);
[V, D]  = eigs(H_inv*P_u_t'*M_t*P_u_t*H_inv', 20);
Dinv    = diag(1./(1-diag(D)));
aprx_Inv  = V*Dinv*V';
%exactMI   = inv(eye(size(H_inv))-KB_uu_inv*P_u_t'*M_t*P_u_t);
exactMI   = inv(eye(size(H_inv))-H_inv*P_u_t'*M_t*P_u_t*H_inv');
norm(aprx_Inv-exactMI,'fro')/norm(exactMI,'fro')
%aprx_Inv = H_inv*aprx_Inv*H_inv';
muH       = H\mu_t;
norm_mu_t = muH'*aprx_Inv*muH;
end
function [norm_mu_t, aprx5] = get_approx_by_mul(mu_t, M_t, KB_uu_inv, P_u_t)
L     = KB_uu_inv* P_u_t'*M_t*P_u_t*KB_uu_inv;
aprx2 = KB_uu_inv + L;
L     = L* P_u_t'*M_t*P_u_t*KB_uu_inv;
aprx3 = aprx2 + L;
L     = L* P_u_t'*M_t*P_u_t*KB_uu_inv;
aprx4 = aprx3 + L;
L     = L* P_u_t'*M_t*P_u_t*KB_uu_inv;
aprx5 = aprx4 + L;
norm_mu_t = mu_t'*aprx5*mu_t;
end
function nu_t = comp_nu_t( R, mu_R_i, ind_t, F_A, F_id, midx_A, n_P_t, z_l, initL, dist_id )
nu_t = zeros(n_P_t,1);
n_l = numel(z_l);
for i= 1:n_l
   id    = initL(i);
   index = (F_id==id);
   ind_i = (F_A == id);
   R_ti  = R(ind_t, ind_i);
   mu_R_ii = mu_R_i{index};
   nu_t  = nu_t + R_ti*mu_R_ii*z_l(i); 
end
end
function P_u_t = get_P_u_t(R, mu_R_i, ind_t, F_A, F_id, midx_A, n_P_t, z_l, initL)
n_l   = numel(z_l);
n_u   = midx_A - n_l; 
P_u_t = zeros(n_P_t, n_u);
j = 1;
for i= 1:midx_A
   id    = F_id(i);
   if(~ismember(id, initL))
      index = (F_id==id);
      ind_i = (F_A == id);
      R_ti  = R(ind_t, ind_i);
      mu_R_ii = mu_R_i{index};
      P_u_t(:,j) = R_ti*mu_R_ii;
      j = j+1;
   end
end
end
function x_k    = it_psd(A_inv, B, m)
tol     = 1e-5;
x_pk    = A_inv*m;
A_invB  = A_inv*B;
normdif = 1000;
while(normdif>tol)
   x_k = A_inv*m + A_invB*x_pk;
   normdif = norm(x_k-x_pk);
   x_pk = x_k;
   disp(normdif);
end

end
