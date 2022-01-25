#include "math.h"
#include "mex.h"

#include "mymatrix_lib.h"

void   print_vec(double *p, mwSize d);
void   print_vec(mwSize *p, mwSize d);
void   print_vec(int *p, mwSize d);
void print_matrix(double *p, mwSize d, mwSize n);
// this function prints an structure in the form of pointer to an array of pointers to arrays         
void print_p_tocells(double * (**p), mwSize n, mwSize* FA_n);
/* This function computes the rbf kernel with the parameter gamma */
double rbf_kernel_compute(double gamma, mwSize d, double *y, double *z);
/* This function computes sum of kernel between each element of distribution A and B
    A is an array of size d times mA and also, b, gamma is the parameter of kernel*/
double rbf_kernel_of_dist(double gamma, mwSize d, mwSize mA, mwSize mB, double *A, double *B);
/* This is the sames as the above function when two distributions are equal */
double rbf_kernel_self_ofdist(double gamma, mwSize d, mwSize mA, double *A);
/* This function computes the value of meanembeding of distribution A at a location x, its a real value*/
double meanembed_funcval_at_x(double gamma, mwSize d, mwSize nA, double *A,  double* x);
// This function computes the values related to a distribution i: stidx_A, enidx_A, nP_i: size of distributions, P_i_ID: id of distribution
void get_dist_info(mwSize *idx_dF_A, mwSize *uF_A, mwSize midx_A, mwSize i, mwSize* stidx_A, mwSize* enidx_A, mwSize* nP_i, mwSize * P_i_ID);
/* This function computes the value of meanembeding of distribution A at each location of instances drawn from distribution itself, 
 * it's a real array in the same dimension as the number of instances drawn from distribution*/
void meanembed_fval_at_all(double gamma, mwSize d, mwSize nA, double *A, double* vec_out);
/* This function computes meanembeding of distributions at each point of each distribution. 
 * It's output is a pointer to an array of pointers to array of embedding of distrributions for each point*/
void meanembed_fval_dist( mwSize d, mwSize nA, double *A, double *F_A, mwSize*uF_A, mwSize *idx_dF_A, mwSize midx_A, double gamma, 
                                double ***p_to_cells, mwSize* FA_map, mwSize *FA_n);
// this function deletes an structure in the form of pointer to an array of pointers to arrays         
void delete_p_to_cells(mwSize midx_A, double ***p_to_cells);
double normdiff_of_dist(double gamma, mwSize d, mwSize mA, mwSize mB, double *A, double *B);
double rbf_emp_kernel(double gamma, mwSize d, mwSize mA, mwSize mB, double *A, double *B);
void rbf_emp_matrix( mwSize d, mwSize nA, double *A, double *F_A, mwSize*uF_A, mwSize *idx_dF_A, mwSize midx_A,  
                               mwSize nB, double *B, double *F_B, mwSize*uF_B, mwSize*idx_dF_B, mwSize midx_B, double gamma, 
                               double* dm, mwSize *FA_map, mwSize*FB_map  );
/* This function computes the R kernel matrix which is a kernel such that the draws from the corresponding 
 * distribution generate distributions which are in embedding kernel space, see paper: Bayesian learning of kernel mean embedding
 * It is a matrix for kernels between any instance in any distributions with any other instance*/
void R_forall_indexed( mwSize d, mwSize nA, double *A, double gamma_r, double* R);
/* This function copies an square submatrix of Matrix R from stidx_A to enidx_A*/
void copy_matrix_part(mwSize nA, mwSize stidx_A, mwSize enidx_A, double*R, double* R_reginv_i);
/* This function adds a regularization term to maindiagonal of matrix R_i*/
void reg_matrix(mwSize nR_i, double* R_i, double regterm);
/* This function extract a submatrix of regarding a distribution from R and then regualrize it using the above function and finaly 
  Inverse the matrix */
void R_reginv_dist(mwSize nA, mwSize stidx_A, mwSize enidx_A, double*R, double* R_reginv_i, double thau);
/* This function computes the above matrix for each matrix*/
void compute_R_reginv_forall( mwSize d, mwSize nA, double *R, double *F_A, mwSize*uF_A, mwSize *idx_dF_A, mwSize midx_A, double thau, 
                                double ***pR_to_cells, mwSize* FA_map, mwSize *FA_n);
void get_FA_n(mwSize nA, double *F_A, mwSize*uF_A, mwSize *idx_dF_A, mwSize midx_A, mwSize *FA_n);
/* This function computes the multiplication of pR_to_cells to p_to_cells for each distribution in A,
 The first term is the R_reginv matrices and the second is the mu_P_*/
void  multiply_mu_P_Rreginv(double*** pR_to_cells, double*** p_to_cells, double*** pResult_to_cells, mwSize midx_A, mwSize *FA_n);
/* This function computes the Kernel matrix between distributions i and j */
double compute_KBayes_ij(mwSize i, mwSize j, mwSize nA, double *R, mwSize  stidx_A_i, mwSize enidx_A_i, mwSize  stidx_A_j, mwSize enidx_A_j,
                                    double* pResult_i, mwSize n_i, double* pResult_j, mwSize n_j );
void comp_mu_R_all( mwSize d, mwSize nA, double *A, double *F_A, mwSize* uF_A, mwSize *idx_dF_A, mwSize midx_A, double gamma, double thau, double* R,
                                                   double *** pResult_to_cells, double*** pR_reginv_to_cells, double *FA_map, mwSize *FA_n);
/* This function computes the matrix KBayes between each distributions and also, 
 * computes and returns matrix R*/
void compute_KBayes( mwSize d, mwSize nA, double *A, double *F_A, mwSize* uF_A, mwSize* idx_dF_A, mwSize midx_A, double gamma, double gamma_r, double thau, 
                                          double *KB, double* R, double *** pResult_to_cells, double*** pR_reginv_to_cells, mwSize *FA_map, mwSize *FA_n);
int  foundin(mwSize item, mwSize* initL, mwSize n_l);
// This function computes nu_t = \sum_{i\in D_l} R^{ti} *mu_R_i*z_l[i]
void comp_nu_t(double*R, const mxArray* mu_R_icell, mwSize stidx_A_t, mwSize enidx_A_t, mwSize nA, mwSize* uF_A, mwSize *idx_dF_A, mwSize midx_A, mwSize n_P_t,
               double*z_l, mwSize initL[], mwSize n_l, double* nu_t, mwSize t);
void get_wnorm_lestimate(double *KB_l, double *KB_uu, double *KB_lu_z_l, double *R, mwSize nA, double *z_l, mwSize initL[], mwSize n_l, mwSize* FA_n, 
                         const mxArray* mu_R_icell, const mxArray* R_reginv_cell, mwSize *uF_A, mwSize *idx_dF_A, mwSize midx_A, double* w_l_norm_t, double tol);
void distance_matrix( mwSize d, mwSize nA, double *A, double *F_A, mwSize*uF_A, mwSize *idx_dF_A, mwSize midx_A,
                                mwSize nB, double *B, double *F_B, mwSize*uF_B, mwSize*idx_dF_B, mwSize midx_B,  double gamma, 
                                double* dm, mwSize *FA_map, mwSize*FB_map  );
void get_wnorm_estimate(double *y_l, mwSize *Lindex, mwSize *Uindex, double *KB, double*KBreg_inv,
                        double *R, mwSize* FA_n, mwSize *F_id, const mxArray* mu_R_icell, const mxArray* R_reginv_cell,
                        mwSize nA, mwSize n_l, mwSize *uF_A, mwSize *idx_dF_A, mwSize midx_A, double* w_norm_t ,double tol);
// VERY IMPORTANT: ORDER OF uF_A AND F_A MUST BE THE SAME ,i.e. DISTRIBUTION uF_A[i] MUST BE THE i-th DISTINCT DISTRIBUTION IN F_A 
// FROM stidx_A = idx_dF_A[i-1] TO enidx_A = idx_dF_A[i]. IN ORDER TO ENFORCE THIS, WE MUST CHECK THIS IN compute_dist_idx. 
// It is better to compute uF_A based on the F_A, so there is no need to get it from the inputs which is a redundancy. 
void distance_matrix_sameset( mwSize d, mwSize nA, double *A, double *F_A, mwSize* uF_A, mwSize *idx_dF_A, mwSize midx_A, 
                              double gamma, double* dm, mwSize *FA_map);
void mydiff(mwSize nF,  double* F, mwSize *dF);
void findnz(mwSize ndF, mwSize* dF, mwSize *max_ndF, mwSize *idx_dF);
// VERY IMPORTANT: ORDER OF uF_A AND F_A MUST BE THE SAME ,i.e. DISTRIBUTION uF_A[i] MUST BE THE i-th DISTRIBUTION IN F_A 
// FROM stidx_A = idx_dF_A[i-1] TO enidx_A = idx_dF_A[i]. IN ORDER TO ENFORCE THIS, WE MUST CHECK THIS IN compute_dist_idx. 
void compute_dist_idx(mwSize nA, double* F_A, mwSize* max_ndF, mwSize *idxF, mwSize *uF);
mwSize get_numberof_dist(mwSize nA, double* F_A);
/* This is the same as function meanembed_fval_dist but without index of distributions*/ 
void meanembed_noindex_dist( mwSize d, mwSize nA, double *A, double *F_A, double gamma, 
                                                   double ***p_to_cells , mwSize *FA_map, mwSize *FA_n);
/* This is the above function except that it doesnot have indices of distributions*/
void R_forall_noindex( mwSize d, mwSize nA, double *A, double *F_A, double gamma_r, double* R);
void compute_KBayes_noindex( mwSize d, mwSize nA, double *A, double *F_A, double gamma, double gamma_r, double thau, 
                                                  double *KB, double* R, double *** pResult_to_cells, double*** pR_reginv_to_cells, double *FA_map, mwSize *FA_n);
/* This is the same function as compute_R_reginv_forall but without indices*/ 
void compute_R_reginv_forall_noindex( mwSize d, mwSize nA, double *R, double *F_A, double thau, 
                                                   double ***pR_to_cells , mwSize *FA_map, mwSize *FA_n);
void distance_matrix_noindex( mwSize d, mwSize nA, double *A, double *F_A, mwSize nB, double *B, double *F_B, double gamma, 
                                                   double* dm , mwSize *FA_map, mwSize*FB_map  );
void distance_matrix_noindex_sameset( mwSize d, mwSize nA, double *A, double *F_A, double gamma, 
                                                   double* dm , mwSize *FA_map);
void rbf_emp_matrix_noindex( mwSize d, mwSize nA, double *A, double *F_A, mwSize nB, double *B, double *F_B, double gamma, 
                                                   double* dm , mwSize *FA_map, mwSize*FB_map  );