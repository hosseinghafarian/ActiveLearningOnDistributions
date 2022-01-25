#include "math.h"
#include "mex.h"
#include "kernel_lib.h"
#include "mymatrix_lib.h"

void   print_vec(double *p, mwSize d){
    mwSize i;
	mexPrintf("[ ");
    for(i=0;i<d;i++)
        mexPrintf("%f,",p[i]);
	mexPrintf("]");
}
void   print_vec(mwSize *p, mwSize d){
    mwSize i;
	mexPrintf("[ ");
    for(i=0;i<d;i++)
        mexPrintf("%d,",p[i]);
	mexPrintf("]");
}
void   print_vec(int *p, mwSize d){
    mwSize i;
	mexPrintf("[ ");
    for(i=0;i<d;i++)
        mexPrintf("%d,",p[i]);
	mexPrintf("]");
}
void print_matrix(double *p, mwSize d, mwSize n){
    mwSize i,j;
    mexPrintf("[");
    for(i=0;i<d;i++){
        for(j=0;j<n;j++)
            mexPrintf("%7.5f  ",p[i+j*d]);
        if(i==d) break;
        mexPrintf("\n");
    }
    mexPrintf("]");
}
// this function prints an structure in the form of pointer to an array of pointers to arrays         
void print_p_tocells(double * (**p), mwSize n, mwSize* FA_n){
 int i;
 mexPrintf(" Print p_to_cells:\n");
 if (!p)
     mexPrintf("p is null\n");
 else{
     if (!(*p))
         mexPrintf("p is null\n");
     else {
     for(i=0;i<n; i++){
         mexPrintf("print cell (%d):",i);
         if ((*p)[i])
             print_vec((*p)[i], FA_n[i]);
         else
             mexPrintf("p is null\n");
     }   
     }
 }
}
/* This function computes the rbf kernel with the parameter gamma */
double rbf_kernel_compute(double gamma, mwSize d, double *y, double *z)
{
    mwSize i;
    double sum, retval, diff;
    /* multiply each element y by x */
    sum = 0;
    for (i=0; i<d; i++) {
        diff = z[i]- y[i];
        sum += diff*diff;
    }
    retval = exp(-0.5*gamma*sum);
    return retval;
}
/* This function computes sum of kernel between each element of distribution A and B
    A is an array of size d times mA and also, b, gamma is the parameter of kernel*/
double rbf_kernel_of_dist(double gamma, mwSize d, mwSize mA, mwSize mB, double *A, double *B){
    mwSize i, j;
    double sum, value;		
    sum = 0;
    for(i = 0; i<mA; i++)
        for(j = 0 ; j<mB;j++){
		   value = rbf_kernel_compute(gamma, d, &A[i*d], &B[j*d]);
		   sum += value;
        }
    return sum;
}
/* This is the sames as the above function when two distributions are equal */
double rbf_kernel_self_ofdist(double gamma, mwSize d, mwSize mA, double *A) {
	mwSize i, j;
	double sum, value,sumrow;
	sum = 0;
	for (i = 0; i < mA; i++) {
        sumrow = 0;
		for (j = i + 1; j < mA; j++) {
			value = rbf_kernel_compute(gamma, d, &A[i*d], &A[j*d]);
			sumrow += value;
		}
        sum += 2*sumrow;
        value = rbf_kernel_compute(gamma, d, &A[i*d], &A[i*d]);
		sum += value;
	}
	return sum;
}
/* This function computes the value of meanembeding of distribution A at a location x, its a real value*/
double meanembed_funcval_at_x(double gamma, mwSize d, mwSize nA, double *A,  double* x) {
    const double tol = 1e-12;
    double value;
    mwSize i;
    value   = 0;
    for(i=0; i < nA;i++){
       value += rbf_kernel_compute(gamma, d, &A[i*d], x);
    }
    if (value <tol){
        value = 0;
    }
    value = value/nA;
    return value;
}
// This function computes the values related to a distribution i: stidx_A, enidx_A, nP_i: size of distributions, P_i_ID: id of distribution
void get_dist_info(mwSize *idx_dF_A, mwSize *uF_A, mwSize midx_A, mwSize i, mwSize* stidx_A, mwSize* enidx_A, mwSize* nP_i, mwSize * P_i_ID){
     if (i == 0) 
         *stidx_A = 0;
     else
         *stidx_A = idx_dF_A[i-1];
     *enidx_A = idx_dF_A[i];
     if (*enidx_A <0 ){
          mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error negative index");
     }
     *nP_i    = *enidx_A - *stidx_A;
     if (*nP_i <= 0) 
         mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error negative length");

     *P_i_ID  = uF_A[i]; 
}
/* This function computes the value of meanembeding of distribution A at each location of instances drawn from distribution itself, 
 * it's a real array in the same dimension as the number of instances drawn from distribution*/
void meanembed_fval_at_all(double gamma, mwSize d, mwSize nA, double *A, double* vec_out){
     mwSize i;
     for(i=0; i< nA; i++ ) 
         vec_out[i] = meanembed_funcval_at_x(gamma, d, nA, A, &A[i*d]);
}
/* This function computes meanembeding of distributions at each point of each distribution. 
 * It's output is a pointer to an array of pointers to array of embedding of distributions for each point*/
void meanembed_fval_dist( mwSize d, mwSize nA, double *A, double *F_A, mwSize*uF_A, mwSize *idx_dF_A, mwSize midx_A, double gamma, 
                                double ***p_to_cells, mwSize *FA_map, mwSize *FA_n){
    const double tol = 1e-12;
    double *X_i;
    double **p;

    mwSize i, stidx_A, enidx_A, nX_i;
   
    p  = new double*[midx_A];
    *p_to_cells = p;
    for(i=0; i < midx_A;i++){
       get_dist_info(idx_dF_A, uF_A, midx_A, i, &stidx_A, &enidx_A, &nX_i, &FA_map[i]);
       X_i       = &A[stidx_A*d];
       p[i]      = new double[nX_i];
       FA_n[i]   = nX_i;
       meanembed_fval_at_all(gamma, d, nX_i, X_i, p[i]);
       if(i==0){
          mexPrintf("\n"); 
          print_vec(p[i], nX_i);
          mexPrintf("\n");
       }
    }    
}
// this function deletes an structure in the form of pointer to an array of pointers to arrays         
void delete_p_to_cells(mwSize midx_A, double ***p_to_cells){
    mwSize i;
    for(i=0; i < midx_A;i++){
       double *p = (*p_to_cells)[i];
       delete (*p_to_cells)[i];
    }
    delete p_to_cells;
}
double normdiff_of_dist(double gamma, mwSize d, mwSize mA, mwSize mB, double *A, double *B){
    if (d<=0){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error nonpositive dimension");
        return -1;
    }
    if ((mA<1) ||(mB<1)){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Zero or neative number of instance! Error.");
        return -1;
    }
    double sAA, sBB, sAB, result;
    sAA = rbf_kernel_self_ofdist(gamma, d, mA, A);
    sAB = rbf_kernel_of_dist(gamma, d, mA, mB, A, B); 
    sBB = rbf_kernel_self_ofdist(gamma, d, mB, B);
    result = sAA/(mA*mA) + sBB/(mB*mB) - 2*sAB/(mA*mB);
    return result;
}
double rbf_emp_kernel(double gamma, mwSize d, mwSize mA, mwSize mB, double *A, double *B){
    if (d<=0){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error nonpositive dimension");
        return -1;
    }
    if ((mA<1) ||(mB<1)){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Zero or neative number of instance! Error.");
        return -1;
    }
    double sAB, result;
    sAB = rbf_kernel_of_dist(gamma, d, mA, mB, A, B); 
    result = sAB/(mA*mB);
    return result;
}
void rbf_emp_matrix( mwSize d, mwSize nA, double *A, double *F_A, mwSize *uF_A, mwSize *idx_dF_A, mwSize midx_A,  
                               mwSize nB, double *B, double *F_B, mwSize *uF_B, mwSize *idx_dF_B, mwSize midx_B, double gamma, 
                               double* dm, mwSize *FA_map, mwSize *FB_map  ){
    const double tol = 1e-12;
    double *X_i, *X_j,value;
    mwSize i,j, stidx_A, stidx_B, enidx_A, enidx_B, nX_i, nX_j;
    for(i=0; i < midx_A;i++){
       get_dist_info(idx_dF_A, uF_A, midx_A, i, &stidx_A, &enidx_A, &nX_i, &FA_map[i]);
       X_i = &A[stidx_A*d]; 
       for(j =0; j<midx_B;j++){
           get_dist_info(idx_dF_B, uF_B, midx_B, j, &stidx_B, &enidx_B, &nX_j, &FB_map[j]);
           X_j = &B[stidx_B*d]; 
           value = rbf_emp_kernel(gamma, d, nX_i, nX_j, X_i, X_j);
           if (value <tol){
               value = 0;
           }
           dm[j*midx_A+i] = value;
       }
    }
}
/* This function computes the R kernel matrix which is a kernel such that the draws from the corresponding 
 * distribution generate distributions which are in embedding kernel space, see paper: Bayesian learning of kernel mean embedding
 * It is a matrix for kernels between any instance in any distributions with any other instance*/
void R_forall_indexed( mwSize d, mwSize nA, double *A, double gamma_r, double* R){
    const double tol = 1e-12;
    double *X_i, *X_j, value;
    mwSize i,j;
    for(i=0; i < nA; i++){
       X_i = &A[i*d];
       
       for(j =0; j<=i;j++){
           X_j = &A[j*d];
           value = rbf_kernel_compute(gamma_r, d, X_i, X_j);
           if (value <tol){
               value = 0;
           }
           R[j*nA+i] = value;
           R[i*nA+j] = value;
       }
    }
}
/* This function copies an square submatrix of Matrix R from stidx_A to enidx_A*/
void copy_matrix_part(mwSize nA, mwSize stidx_A, mwSize enidx_A, double*R, double* R_reginv_i){
    mwSize nX_i= enidx_A-stidx_A;
    for(mwSize i= 0;i<nX_i;i++)
        for(mwSize j= 0;j<nX_i;j++)
            R_reginv_i[j*nX_i+i ] = R[(j+stidx_A)*nA+(i+stidx_A)];
}
void copy_matrix_part_rect(mwSize nA, mwSize stidx_A_i, mwSize enidx_A_i, mwSize stidx_A_j, mwSize enidx_A_j, double*R, double* R_copy){
    mwSize nX_i= enidx_A_i-stidx_A_i;
    mwSize nX_j= enidx_A_j-stidx_A_j;
    for(mwSize i= 0;i<nX_i;i++)
        for(mwSize j= 0;j<nX_j;j++)
            R_copy[j*nX_i+i ] = R[(j+stidx_A_j)*nA+(i+stidx_A_i)];
}
/* This function adds a regularization term to maindiagonal of matrix R_i*/
void reg_matrix(mwSize nR_i, double* R_i, double regterm){
    for(mwSize j= 0;j<nR_i;j++)
        R_i[j*nR_i+j ] = R_i[j*nR_i+j ] + regterm;
}
/* This function extract a submatrix of regarding a distribution from R and then regualrize it using the above function and finaly 
  Inverse the matrix */
void R_reginv_dist(mwSize nA, mwSize stidx_A, mwSize enidx_A, double*R, double* R_reginv_i, double thau){
    if (stidx_A >= enidx_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: wrong start and end index for distribution in R_reginv_dist");
    mwSize nX_i = enidx_A-stidx_A;
    copy_matrix_part(nA, stidx_A, enidx_A, R, R_reginv_i);
    if (nX_i<1) nX_i = 1;
    double regterm = thau/nX_i;
    reg_matrix(nX_i, R_reginv_i, regterm);
    inverseMatrix(nX_i, R_reginv_i, R_reginv_i, 0);
}
/* This function computes the above matrix for each matrix*/
void compute_R_reginv_forall( mwSize d, mwSize nA, double *R, double *F_A, mwSize *uF_A, mwSize *idx_dF_A, mwSize midx_A, double thau, 
                                double ***pR_to_cells, mwSize *FA_map, mwSize *FA_n){
    const double tol = 1e-12;

    double **p;
    mwSize i, stidx_A, enidx_A, nX_i;
   
    p = *pR_to_cells;

    for(i=0; i < midx_A;i++){
       get_dist_info(idx_dF_A, uF_A, midx_A, i, &stidx_A, &enidx_A, &nX_i, &FA_map[i]);
       FA_n[i] = nX_i;
       R_reginv_dist(nA, stidx_A, enidx_A, R, p[i], thau);
    }    
}
void get_FA_n(mwSize nA, double *F_A, mwSize *uF_A, mwSize *idx_dF_A, mwSize midx_A, mwSize *FA_n){
    mwSize i, stidx_A, enidx_A;
    mwSize dist_id;
    stidx_A = 0;
    for(i=0; i < midx_A;i++){
       get_dist_info(idx_dF_A, uF_A, midx_A, i, &stidx_A, &enidx_A, &FA_n[i], &dist_id);
    }    
}
/* This function computes the multiplication of pR_to_cells to p_to_cells for each distribution in A,
 The first term is the R_reginv matrices and the second is the mu_P_*/
void  multiply_mu_P_Rreginv(double*** pR_to_cells, double*** p_to_cells, double*** pResult_to_cells, mwSize midx_A, mwSize *FA_n){
    
    //double **p  = new double*[midx_A];
    double **p    = *pResult_to_cells;
    double **pR   = *pR_to_cells;
    double **p_mu = *p_to_cells;
    for(mwSize i = 0; i < midx_A ; i++){
       mwSize nX_i = FA_n[i];
       //p[i] = new double[nX_i];
       matrix_multiply(nX_i, nX_i, nX_i, 1, pR[i], p_mu[i], p[i]);
    }
}
/* This function computes the Kernel matrix between distributions i and j */
double compute_KBayes_ij(mwSize t, mwSize v, mwSize nA, double *R, mwSize  stidx_A_i, mwSize enidx_A_i, mwSize  stidx_A_j, mwSize enidx_A_j,
                                    double* pResult_i, mwSize n_i, double* pResult_j, mwSize n_j ){
    
    double result;// the initial value must be changed 
    mwSize n_A_i = enidx_A_i-stidx_A_i;
    mwSize n_A_j = enidx_A_j-stidx_A_j;
    double *mat= new double[n_A_j*n_A_i];
    double *vec = new double[n_A_i];
    // mexPrintf("P_i: (%d, %d) P_j: (%d, %d)\n",  stidx_A_i, enidx_A_i, stidx_A_j, enidx_A_j);
    // mwSize leadingdim_A = nA;
    // double *pR_start = R + stidx_A_j*nA + stidx_A_i;// check if it is correct?
    // mwSize dimsubA_1 = enidx_A_i-stidx_A_i;
    // mwSize dimsubA_2 = enidx_A_j-stidx_A_j;
    // compute R_ij*pResj = R_ij*(R_j+tha/n_j I)^{-1} mu_P_X_j
    // double *pRijResj = new double[dimsubA_1];
    // submatrix_multiply(dimsubA_1, dimsubA_2, n_j, 1, pR_start, leadingdim_A, pResult_j, pRijResj) ;
    for(mwSize i=0;i<(enidx_A_i-stidx_A_i);i++){
        for(mwSize j=0;j<(enidx_A_j-stidx_A_j);j++){
           mat[j*n_A_i+i] = R[(j+stidx_A_j)*nA+i+stidx_A_i];
           if (mat[j*n_A_i+i] >1)
               mexPrintf("R_ij : R(%d,%d)=R_(%d):%f\n", i+stidx_A_i, j+stidx_A_j, (j+stidx_A_j)*nA+i+stidx_A_i, mat[j*n_A_i+i]); 
        }
    }
//     if (stidx_A_i==0)
//         print_vec(mat, n_A_i*n_A_j);
    for(mwSize i=0;i<(enidx_A_i-stidx_A_i);i++){
        vec[i] = 0;
        for(mwSize j=0;j<(enidx_A_j-stidx_A_j);j++){
           vec[i] += mat[j*n_A_i+i]*pResult_j[j];
        }
    }
    double val = 0.0;
    
    for(mwSize i=0;i<n_A_i;i++){
        val += vec[i]*pResult_i[i];
    }
    double one = 1.0;
    double zero = 0.0;
    double *pRijResj = new double[enidx_A_i-stidx_A_i];
    for(mwSize ti=0; ti< n_A_i; ti++)
        pRijResj[ti] = 0.0;
    
    result = 0.0;
    submat_mult_alphbeta(stidx_A_i, enidx_A_i, stidx_A_j, enidx_A_j, nA, nA, 1, R, 
                          n_j, one, pResult_j, zero, pRijResj);
    matrix_multiply(1, enidx_A_i-stidx_A_i, enidx_A_i-stidx_A_i, 1,  pResult_i, pRijResj, &result);
    delete pRijResj;
    if (result>1 && v==0){//||( val!= result && stidx_A_i==0 )){
        mexPrintf("Error: result are not correct\n");
        print_matrix(mat, enidx_A_i-stidx_A_i, enidx_A_j-stidx_A_j);
        mexPrintf("\n stidx_A_i:%d, enidx_A_i:%d,", stidx_A_i, enidx_A_i);
        mexPrintf("stidx_A_j:%d, enidx_A_j:%d\n", stidx_A_j, enidx_A_j);
    }
//     if ( val== result && j==0 )
//         mexPrintf("KB(%d,%d): %f, Ok\n", i,j,val);
    if ( val!= result && v==0 )
        mexPrintf("KB(%d,%d): %f, %f  not Ok\n", t,v,val, result);
    delete mat;
    delete vec;
    return result;
}
void comp_mu_R_all( mwSize d, mwSize nA, double *A, double *F_A, mwSize* uF_A, mwSize *idx_dF_A, mwSize midx_A, double gamma, double thau, double* R,
                                                   double *** pResult_to_cells, double*** pR_reginv_to_cells, mwSize *FA_map, mwSize *FA_n){
    // compute the Rregiv for all distributions
    compute_R_reginv_forall(d, nA, R, F_A, uF_A, idx_dF_A, midx_A, thau, pR_reginv_to_cells, FA_map, FA_n);
    double **pR = *pR_reginv_to_cells;
    // compute the mu_p_X_i for all distributions. 
    double*** p_to_cells = (double ***)new double*; 
    meanembed_noindex_dist( d, nA, A, F_A, gamma, p_to_cells, FA_map, FA_n);
    double **pMu_P_X = *p_to_cells;
    // compute mu_R_i= Rregiv_i*mu_p_X_i = (R_i + thau/n_i)^{-1} mu_p_X_i;
    multiply_mu_P_Rreginv(pR_reginv_to_cells, p_to_cells, pResult_to_cells, midx_A, FA_n);

    delete_p_to_cells(midx_A, p_to_cells);
}
/* This function computes the matrix KBayes between each distributions and also, 
 * computes and returns matrix R*/
void compute_KBayes( mwSize d, mwSize nA, double *A, double *F_A, mwSize* uF_A, mwSize* idx_dF_A, mwSize midx_A, double gamma, double gamma_r, double thau, 
                                          double *KB, double* R, double *** pResult_to_cells, double*** pR_reginv_to_cells, mwSize *FA_map, mwSize *FA_n){
      
    if ((d<1)||(nA<1))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error, zero or negative columns or dimension");
    
    if (!idx_dF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: indices of distributions didnot computed");
    // compute the R matrix 
    R_forall_indexed(d, nA, A, gamma_r, R);
    // compute mu_R_i= Rregiv_i*mu_p_X_i = (R_i + thau/n_i)^{-1} mu_p_X_i;
    
    comp_mu_R_all( d, nA, A, F_A, uF_A, idx_dF_A, midx_A, gamma, thau, R, pResult_to_cells, pR_reginv_to_cells, FA_map, FA_n);
    //print_vec((*pResult_to_cells)[1], FA_n[1]);
    double **pResult = *pResult_to_cells;
    // compute KBayes
    double val;
    mwSize n_i, n_j;
    mwSize stidx_A_i,stidx_A_j,sti,stj;
    mwSize enidx_A_i,enidx_A_j,eni,enj;
    //print_vec(idx_dF_A, 15);
    //mexPrintf("\n");
    for(mwSize i=0; i < midx_A;i++){
       get_dist_info(idx_dF_A, uF_A, midx_A, i, &stidx_A_i, &enidx_A_i, &n_i, &FA_map[i]);
       sti = stidx_A_i; eni = enidx_A_i;
       //mexPrintf("stidx_A_i:%d, enidx_A_i:%d\n",stidx_A_i, enidx_A_i);   
       for(mwSize j =0; j<=i ;j++){       
           get_dist_info(idx_dF_A, uF_A, midx_A, j, &stidx_A_j, &enidx_A_j, &n_j, &FA_map[j]);
           stj = stidx_A_j; enj = enidx_A_j;
//            mexPrintf("i:%d, j:%d\n",i,j);
           if (n_j >= n_i){ // first multiply larger dimension to make a smaller vector.
               val = compute_KBayes_ij(i,j, nA, R, sti, eni, stj, enj,
                                                 pResult[i], n_i, pResult[j], n_j);
           }
           else{
               val = compute_KBayes_ij(i,j, nA, R, stj, enj, sti, eni, 
                                                 pResult[j], n_j, pResult[i], n_i);
           }
//            if(i==4 && j==0){
//               mexPrintf("stidx_A_i:%d, enidx_A_i:%d, stidx_A_j:%d, enidx_A_j:%d,  val:%f\n", stidx_A_i, enidx_A_i, stidx_A_j, enidx_A_j, val);   
//            }
           KB[j*midx_A+i] = val;
           KB[i*midx_A+j] = val;
           stidx_A_j = enidx_A_j;
       }
       stidx_A_i = enidx_A_i;
    }
}
int  foundin(mwSize item, mwSize* initL, mwSize n_l){     
     for(mwSize j=0; j<n_l;j++)
        if( initL[j]==item)
           return j;
     return -1;
}
int  cmpvec(double *v1, double*v2, mwSize d){
     double tol = 0.000000001;
     for(mwSize i=0; i<d;i++)
         if(abs(v1[i]-v2[i])>tol){
             mexPrintf("not Equal at v1[%d] = %f, v2[%d] = %f\n", i,v1[i],i, v2[i]); 
             return 0;
         }
     return 1;
}
// This function computes nu_t = \sum_{i\in D_l} R^{ti} *mu_R_i*z_l[i]
void comp_nu_t(double*R, const mxArray* mu_R_icell, mwSize stidx_A_t, mwSize enidx_A_t, mwSize nA, mwSize* uF_A, mwSize *idx_dF_A, mwSize midx_A, mwSize n_P_t,
               double*z_l, mwSize initL[], mwSize n_l, double* nu_t, mwSize t){
    
    for(mwSize i=0; i<n_P_t; i++) nu_t[i] = 0;
    mwSize dist_id, ind_l;
    mwSize i, stidx_A_i, enidx_A_i;
    double beta  = 1.0;
    double alpha;
    double sumf = 0;
    mwSize n_i;
    for(i=0; i < midx_A;i++){
        get_dist_info(idx_dF_A, uF_A, midx_A, i, &stidx_A_i, &enidx_A_i, &n_i, &dist_id); 
        ind_l = foundin(dist_id, initL, n_l);
        if(ind_l!=-1)sumf = sumf+1;
        if (ind_l!=-1){
           const   mxArray* cell_el_ptr = mxGetCell(mu_R_icell, i);
           double* mu_R_i   = mxGetPr(cell_el_ptr);
           mwSize  n_mu_R_i = mxGetM(cell_el_ptr);
           alpha = z_l[ind_l];
           mwSize dt = enidx_A_t-stidx_A_t;
           mwSize ni = enidx_A_i-stidx_A_i;
           submat_mult_alphbeta(stidx_A_t, enidx_A_t, stidx_A_i, enidx_A_i, nA, nA, alpha, R, 
                                n_mu_R_i, 1, mu_R_i, beta, nu_t);
       }
    }
}
void comp_norm_nu_Rreginv(mwSize midx_A, mwSize* FA_n, mwSize *uF_A, mwSize *idx_dF_A, double**nuarray, const mxArray* R_reginv_cell, double *norm_nu){
    mwSize i, stidx_A_i, enidx_A_i, dist_id;
    mwSize n_i;
    for(i=0; i < midx_A;i++){
        get_dist_info(idx_dF_A, uF_A, midx_A, i, &stidx_A_i, &enidx_A_i, &n_i, &dist_id); 
        const   mxArray* cell_el_ptr = mxGetCell(R_reginv_cell, i);
        double* R_reginv_i = mxGetPr(cell_el_ptr);
        mwSize  n_Rreginv_i= mxGetM(cell_el_ptr);
        if (n_i!= n_Rreginv_i)
            mexErrMsgIdAndTxt("MATLAB:matrixMultiply:matchdims",
                "Inner dimensions of matrix multiply do not match in comp_norm_nu_Rreginv");
        norm_nu[i] = norm_with_respect(n_i, nuarray[i], R_reginv_i);
    }
}
// The following function computes result = P_u_t*M_t*nu_t
void mult_P_M_u_t(double *P_u_t, double* M_t, double *nu_t, mwSize n_P_t, mwSize n_u, double * result){
    
    
}
void comp_P_t_vecs(double* R, const mxArray* mu_R_icell, 
                   mwSize stidx_A_t, mwSize enidx_A_t, mwSize stidx_A_i, mwSize enidx_A_i, 
                   mwSize  nA, double* P_t_vecs, mwSize i){
   const   mxArray* cell_el_ptr = mxGetCell(mu_R_icell, i);
   double* mu_R_i   = mxGetPr(cell_el_ptr);
   mwSize  n_mu_R_i = mxGetM(cell_el_ptr);
//    mexPrintf("i in P_t_vecs:%d\n", i);
//    mexPrintf("n_mu_R_i in P_t_vecs:%d\n", n_mu_R_i);
//    mexPrintf("enidx - stidx in P_t_vecs:%d\n", enidx_A_i-stidx_A_i);
   submat_mult_alphbeta(stidx_A_t, enidx_A_t, stidx_A_i, enidx_A_i, nA, nA, 1.0, R, 
                        n_mu_R_i, 1, mu_R_i, 0, P_t_vecs);    
}
void get_P_u_t(double*R, const mxArray* mu_R_icell, mwSize stidx_A_t, mwSize enidx_A_t, mwSize nA, mwSize* uF_A, mwSize *idx_dF_A, mwSize midx_A, 
               mwSize initL[], mwSize n_l, double* P_t_u, mwSize n_P_t){
    
    mwSize dist_id, ind_l;
    mwSize i, j, stidx_A_i, enidx_A_i;
    mwSize n_i;
    double *P_t_vecs= P_t_u;
    for(j=0, i=0; i < midx_A;i++){
        get_dist_info(idx_dF_A, uF_A, midx_A, i, &stidx_A_i, &enidx_A_i, &n_i, &dist_id); 
        ind_l = foundin(dist_id, initL, n_l);
        if (ind_l==-1){ // if notfound
            //mexPrintf("ind_l:%d\n",ind_l);
            // compute R^{ti} *mu_R_i and put it in jth column in P_t_u
            comp_P_t_vecs(R, mu_R_icell, stidx_A_t, enidx_A_t, stidx_A_i, enidx_A_i, nA, P_t_vecs, i);
            P_t_vecs = P_t_vecs+ n_P_t;
            j++;
       }
    }
}
// void get_wnorm_lestimate(double *KB_l, double *KB_uu, double *KB_lu_z_l, double *R, mwSize nA, double *z_l, mwSize initL[], mwSize n_l, mwSize* FA_n, 
//                          const mxArray* mu_R_icell, const mxArray* R_reginv_cell, mwSize *uF_A, mwSize *idx_dF_A, mwSize midx_A, double* w_l_norm_t ){
//     
//     mwSize stidx_A_t, enidx_A_t;
//     mwSize n_P_t, dist_id;    
//     
//     double * norm_nu  = new double[midx_A];
//     //comp_norm_nu_Rreginv(midx_A, FA_n, uF_A, idx_dF_A, nu_array, R_reginv_cell, norm_nu);
//     double z_lKB_lz_l = norm_with_respect(n_l, z_l, KB_l);
//     double norm_pH, norm_mu_t;
//     mwSize n_u = midx_A-n_l;
//     double *mu_t,*nu_t, *p_h_kbz_mu_t, *temp, *temp2, *tempmu_t;
//     double * norm_KB_lu_z_mu_t = new double[midx_A];
//     for(mwSize t=0; t< midx_A; t++){
//         get_dist_info(idx_dF_A, uF_A, midx_A, t, &stidx_A_t, &enidx_A_t, &n_P_t, &dist_id); 
//         nu_t         = new double[n_P_t];
//         double*P_u_t = new double[n_u*n_P_t]; // it's a matrix with n_u columns and n_P_t rows
//         comp_nu_t(R, mu_R_icell, stidx_A_t, enidx_A_t, 
//                   nA, uF_A, idx_dF_A, midx_A, n_P_t, z_l, initL, n_l, nu_t, t);
//         
//         get_P_u_t(R, mu_R_icell, stidx_A_t, enidx_A_t, nA, uF_A, idx_dF_A, midx_A, initL, n_l, P_u_t, n_P_t);
// 
//         tempmu_t     = new double[n_P_t];
//         mu_t         = new double[n_u];
//         
//         const   mxArray* cell_el_ptr = mxGetCell(R_reginv_cell, t);
//         double* M_t  = mxGetPr(cell_el_ptr);
//         mwSize  n_M_t= mxGetM(cell_el_ptr);
//         norm_nu[t]   = norm_with_respect(n_M_t, nu_t, M_t); 
//         // comp mu_t        
//         matrix_multiply(n_P_t, n_P_t, n_P_t, 1, M_t, nu_t, tempmu_t); 
//         matrix_multiply(n_P_t,   n_u, n_P_t, 1, P_u_t, 1, tempmu_t, 0, mu_t);
//         delete tempmu_t;
//         // comp Matrix for inverse KP_TM_TP= \{(K^B_{uu})-{P_t^{u}}^\mathsf{T} M^t P_t^{u}\}
//         double *mu_tInvKP = new double[n_u];
//         for(mwSize j=0; j<n_u;j++){ 
//             mu_t[j]      = KB_lu_z_l[j]- mu_t[j]; 
//             mu_tInvKP[j] = mu_t[j];
//         }
//         temp            = new double[n_P_t*n_u];        
//         double*KP_TM_TP = new double[n_u*n_u];
//         
//         matrix_multiply(n_P_t, n_P_t, n_P_t, n_u, M_t, P_u_t, temp);
//         
//         matrix_multiply(n_P_t, n_u,   n_P_t, n_u, P_u_t, 1, temp, 0, KP_TM_TP);
//         
//         for(mwSize i=0;i<n_u*n_u;i++) KP_TM_TP[i] = KB_uu[i]- KP_TM_TP[i];
//         delete temp;
//         //  solve linear system KP_TM_TP*x= mu_t, mu_tInvKP = x;
//         if(solve_linear(n_u, KP_TM_TP, mu_tInvKP))
//             mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error: cannot solve linear system");
//         norm_pH = 0.0;
//         for(mwSize i=0;i<n_u;i++) norm_pH += mu_tInvKP[i]*mu_t[i];
//         delete KP_TM_TP;
//         
//         norm_KB_lu_z_mu_t[t] = norm_mu_t + norm_pH;
//         
//         delete mu_tInvKP;
//         delete P_u_t;
//         delete nu_t;
//         delete mu_t;
//     }
//     
//     for(mwSize t=0; t<midx_A; t++){
//        w_l_norm_t[t] = z_lKB_lz_l - norm_nu[t] - norm_KB_lu_z_mu_t[t];
//     }
//     
//     delete norm_nu;
//     delete norm_KB_lu_z_mu_t;
// }
void get_P_t(double*R, const mxArray* mu_R_icell, mwSize stidx_A_t, mwSize enidx_A_t, mwSize nA, mwSize* uF_A, mwSize *idx_dF_A, mwSize midx_A, 
             double* P_t_u, mwSize n_P_t){
    
    mwSize dist_id, ind_l;
    mwSize i, j, stidx_A_i, enidx_A_i;
    mwSize n_i;
    double *P_t_vecs= P_t_u;
    for(j=0, i=0; i < midx_A;i++){
        get_dist_info(idx_dF_A, uF_A, midx_A, i, &stidx_A_i, &enidx_A_i, &n_i, &dist_id); 
        // compute R^{ti} *mu_R_i and put it in jth column in P_t_u
        comp_P_t_vecs(R, mu_R_icell, stidx_A_t, enidx_A_t, stidx_A_i, enidx_A_i, nA, P_t_vecs, i);
        P_t_vecs = P_t_vecs+ n_P_t;
        j++;
    }
}
void submatrixcopy(double *source, mwSize sourcecolsize, double*dest, mwSize *Rind, mwSize *Cind, mwSize nR, mwSize nC){
     for(mwSize i=0; i<nR;i++)
         for(mwSize j=0; j<nC;j++)
             dest[j*nR+i] = source[sourcecolsize*(Cind[j])+(Rind[i])];
    
}
void get_wnorm_estimate(double *y_l, mwSize *Lindex, mwSize *Uindex, double *KB, double*KBreg_inv,
                        double *R, mwSize* FA_n, mwSize *F_id, const mxArray* mu_R_icell, const mxArray* R_reginv_cell,
                        mwSize nA, mwSize n_l, mwSize *uF_A, mwSize *idx_dF_A, mwSize midx_A, double* w_norm_t , double tol){
    
    mwSize stidx_A_t, enidx_A_t;
    mwSize n_P_t, dist_id;    
    mwSize n_u = midx_A - n_l ;
//     mexPrintf("\n n_l:%d, n_u:%d\n", n_l, n_u);
    for(mwSize t=0; t< midx_A; t++){
        get_dist_info(idx_dF_A, uF_A, midx_A, t, &stidx_A_t, &enidx_A_t, &n_P_t, &dist_id); 
        

        double*P_t = new double[midx_A*n_P_t]; // it's a matrix with n_u columns and n_P_t rows
        
        get_P_t(R, mu_R_icell, stidx_A_t, enidx_A_t, nA, uF_A, idx_dF_A, midx_A, P_t, n_P_t);

        const   mxArray* cell_el_ptr = mxGetCell(R_reginv_cell, t);
        double* M_t  = mxGetPr(cell_el_ptr);
        mwSize  n_M_t= mxGetM(cell_el_ptr);

        // comp P_t'*M_t*P_t        
        double *mtpt = new double[midx_A* n_P_t];  
        double *L_t  = new double[midx_A*midx_A];
        matrix_multiply(n_P_t, n_P_t, n_P_t, midx_A, M_t, P_t, mtpt); 
        matrix_multiply(n_P_t,midx_A, n_P_t, midx_A, P_t, 1, mtpt, 0, L_t);
        
        //E_t is now in L_t
        delete P_t;
        delete mtpt;
        // comp Matrix for inverse KP_TM_TP= \{(K^B_{uu})-{P_t^{u}}^\mathsf{T} M^t P_t^{u}\}
        for(mwSize j=0; j<midx_A*midx_A;j++){ 
            L_t[j] = KB[j] - L_t[j]; 
        }
        double *E_t = new double[midx_A*midx_A];
        matrix_multiply(midx_A, midx_A, midx_A, midx_A, KBreg_inv, L_t, E_t);
        matrix_multiply(midx_A, midx_A, midx_A, midx_A, E_t, KBreg_inv, L_t); 
        delete E_t;
//         if(t==0)
//             print_matrix(L_t, midx_A, midx_A);
                
        double *L_ul_t = new double[n_u*n_l];
        double *h_t    = new double[n_u];
        submatrixcopy(L_t, midx_A, L_ul_t, Uindex, Lindex, n_u, n_l);
        matrix_multiply(n_u, n_l, n_l, 1, L_ul_t, y_l, h_t);
        delete L_ul_t;
        
        double *L_uu_t = new double[n_u*n_u];
        double *m_t   = new double[n_u];
        for(mwSize j=0;j<n_u;j++) m_t[j] = h_t[j];
        
//         if(t==0){mexPrintf("\n");
//             print_vec(m_t, n_u);
//         }
        submatrixcopy(L_t, midx_A, L_uu_t, Uindex, Uindex, n_u, n_u);
        // regualrize L_uu_t to avoid ill conditioning
        for(mwSize i=0; i<n_u;i++)
             L_uu_t[i*n_u+i] += tol; 
//         if(t==0){mexPrintf("\n");
//            print_matrix(L_uu_t, n_u, n_u);
//         }
        int info = solve_linear(n_u, L_uu_t, m_t);
//         if(t==0)
//            print_vec(m_t, n_u);
        //if(t==0)
            //mexPrintf("\n info:%d\n",info);
        delete L_uu_t;
        if(info){
            delete m_t;
            delete h_t;
            mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: cannot solve linear system");
        }
        double m_th_t = 0.0; 

        for(mwSize j=0;j<n_u;j++) m_th_t += m_t[j] * h_t[j];
        delete m_t;
        delete h_t;
        
        double *L_ll_t = new double[n_l*n_l];
        double *h_l    = new double[n_l];
        submatrixcopy(L_t, midx_A, L_ll_t, Lindex, Lindex, n_l, n_l);
        delete L_t;
        matrix_multiply(n_l, n_l, n_l, 1, L_ll_t, y_l, h_l);
        delete L_ll_t;
        double y_lh_l = 0.0;
        for(mwSize j=0;j<n_l;j++) y_lh_l += h_l[j] * y_l[j];
        delete h_l;
//         mexPrintf("y_lh_l - m_th_t:%f-%f\n", y_lh_l,m_th_t);
        w_norm_t[t] = y_lh_l - m_th_t;    
    }
}
void distance_matrix( mwSize d, mwSize nA, double *A, double *F_A, mwSize*uF_A, mwSize *idx_dF_A, mwSize midx_A,
                                mwSize nB, double *B, double *F_B, mwSize*uF_B, mwSize*idx_dF_B, mwSize midx_B,  double gamma, 
                                double* dm, mwSize *FA_map, mwSize *FB_map  ){
    const double tol = 1e-12;
    double *X_i, *X_j,value;
    mwSize i,j, stidx_A, stidx_B, enidx_A, enidx_B, nX_i, nX_j;
    for(i=0; i < midx_A;i++){
       get_dist_info(idx_dF_A, uF_A, midx_A, i, &stidx_A, &enidx_A, &nX_i, &FA_map[i]); 
       X_i = &A[stidx_A*d]; 
       for(j =0; j<midx_B;j++){
           get_dist_info(idx_dF_B, uF_B, midx_B, j, &stidx_B, &enidx_B, &nX_j, &FB_map[j]);
           X_j = &B[stidx_B*d];            
           value = normdiff_of_dist(gamma, d, nX_i, nX_j, X_i, X_j);
           if (value <tol){
               value = 0;
           }
           dm[j*midx_A+i] = value;
       }
    }
}
// VERY IMPORTANT: ORDER OF uF_A AND F_A MUST BE THE SAME ,i.e. DISTRIBUTION uF_A[i] MUST BE THE i-th DISTINCT DISTRIBUTION IN F_A 
// FROM stidx_A = idx_dF_A[i-1] TO enidx_A = idx_dF_A[i]. IN ORDER TO ENFORCE THIS, WE MUST CHECK THIS IN compute_dist_idx. 
// It is better to compute uF_A based on the F_A, so there is no need to get it from the inputs which is a redundancy. 
void distance_matrix_sameset( mwSize d, mwSize nA, double *A, double *F_A, mwSize* uF_A, mwSize *idx_dF_A, mwSize midx_A, 
                              double gamma, double* dm, mwSize *FA_map){
    const double tol = 1e-12;
    double *X_i, *X_j,value;
    mwSize i,j, stidx_A, stidx_B, enidx_A, enidx_B, nX_i, nX_j;
    
    for(i=0; i < midx_A;i++){
       get_dist_info(idx_dF_A, uF_A, midx_A, i, &stidx_A, &enidx_A, &nX_i, &FA_map[i]); 
       //mexPrintf("dist_id: %d, n_P_i:%d, stidx_A:%d, enidx_A:%d\n", FA_map[i], nX_i, stidx_A, enidx_A);
       X_i = &A[stidx_A*d]; 
       dm[i*midx_A+i] = 0;
       for(j =i+1; j<midx_A;j++){
           get_dist_info(idx_dF_A, uF_A, midx_A, j, &stidx_B, &enidx_B, &nX_j, &FA_map[j]); 
           X_j = &A[stidx_B*d]; 
           value = normdiff_of_dist(gamma, d, nX_i, nX_j, X_i, X_j);
           if (value <tol){
               value = 0;
           }
           dm[i*midx_A+j] = value;
           dm[j*midx_A+i] = value;
       }
    }
}
void mydiff(mwSize nF, double* F, mwSize *dF){
    mwSize i;
    for(i=0;i<nF-1;i++)
        dF[i] = mwSize(F[i+1]- F[i]);
    return;
}
void findnz(mwSize ndF, mwSize* dF, mwSize *max_ndF, mwSize *idx_dF){
    mwSize i,j;
    *max_ndF = 0;
    j=0;
    for(i=0;i<ndF;i++)
        if (dF[i]){
            idx_dF[j] = i+1; j++; *max_ndF = j;
        }
}
// VERY IMPORTANT: ORDER OF uF_A AND F_A MUST BE THE SAME ,i.e. DISTRIBUTION uF_A[i] MUST BE THE i-th DISTRIBUTION IN F_A 
// FROM stidx_A = idx_dF_A[i-1] TO enidx_A = idx_dF_A[i]. IN ORDER TO ENFORCE THIS, WE MUST CHECK THIS IN compute_dist_idx. 
void compute_dist_idx(mwSize nA, double* F_A, mwSize* max_ndF, mwSize *idxF, mwSize *uF){
    mwSize *tmp_idxF = new mwSize[nA];
    if (!tmp_idxF)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    //double *uF       = new double[nA];
    mydiff(nA, F_A, tmp_idxF);
//     mexPrintf("\n we reached dist_idx\n");
//     print_vec(tmp_idxF, nA-1);
    findnz(nA-1,tmp_idxF,max_ndF,idxF);
    if ((*max_ndF)> nA-1)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Too many nonzeros");
    idxF[*max_ndF]=nA;
    (*max_ndF)++;
    uF[0] = (mwSize)F_A[0];
    for(mwSize i=1; i< (*max_ndF); i++){
        mwSize nxStart = idxF[i-1]; 
        uF[i] = (mwSize)F_A[nxStart];
    }
    delete tmp_idxF;
}
mwSize get_numberof_dist(mwSize nA, double* F_A){
    mwSize *tmp_idxF = new mwSize[nA];
    mwSize *idxF     = new mwSize[nA];
    mwSize max_ndF;
    if (!tmp_idxF)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    //double *uF       = new double[nA];
    mydiff(nA, F_A, tmp_idxF);
    findnz(nA-1,tmp_idxF,&max_ndF,idxF);
    if ((max_ndF)> nA-1)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Too many nonzeros");
    idxF[max_ndF] = nA;
    max_ndF++;
    delete tmp_idxF;
    delete idxF;
    return max_ndF;
}
/* This is the same as function meanembed_fval_dist but without index of distributions*/ 
void meanembed_noindex_dist( mwSize d, mwSize nA, double *A, double *F_A, double gamma, 
                                                   double ***p_to_cells , mwSize *FA_map, mwSize *FA_n){
    mwSize *idx_dF_A, *uF_A;
    mwSize midx_A; 
    if ((d<1)||(nA<1))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error, zero or negative columns or dimension");
    
    idx_dF_A = new mwSize[nA];//(double*)mxCalloc(nA, sizeof(double));
    uF_A     = new mwSize[nA];//(double*)mxCalloc(nA, sizeof(double));
    if (!idx_dF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    
    compute_dist_idx(nA, F_A, &midx_A, idx_dF_A, uF_A ); 

    meanembed_fval_dist(d, nA, A, F_A, uF_A, idx_dF_A, midx_A, gamma, p_to_cells, FA_map, FA_n);

    delete idx_dF_A;//mxFree(idx_dF_A);
    delete uF_A;
}
/* This is the above function except that it doesnot have indices of distributions*/
void R_forall_noindex( mwSize d, mwSize nA, double *A, double *F_A, double gamma_r, double* R){
    if ((d<1)||(nA<1))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error, zero or negative columns or dimension");
    R_forall_indexed(d, nA, A, gamma_r, R);
}
void compute_KBayes_noindex( mwSize d, mwSize nA, double *A, double *F_A, double gamma, double gamma_r, double thau, 
                                                  double *KB, double* R, double *** pResult_to_cells, double*** pR_reginv_to_cells, mwSize *FA_map, mwSize *FA_n){  
    mwSize *idx_dF_A,*uF_A;
    mwSize midx_A; 
    if ((d<1)||(nA<1))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error, zero or negative columns or dimension");
    
    idx_dF_A = new mwSize[nA];//(double*)mxCalloc(nA, sizeof(double));
    uF_A     = new mwSize[nA];//(double*)mxCalloc(nA, sizeof(double));
    if (!idx_dF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    
    compute_dist_idx(nA, F_A, &midx_A, idx_dF_A, uF_A ); 
    // compute the R matrix 
    compute_KBayes(d, nA, A, F_A, uF_A, idx_dF_A, midx_A, gamma, gamma_r, thau, KB, R, pResult_to_cells, pR_reginv_to_cells,  FA_map, FA_n);
    
    delete idx_dF_A;//mxFree(idx_dF_A);
    delete uF_A;
}
/* This is the same function as compute_R_reginv_forall but without indices*/ 
void compute_R_reginv_forall_noindex( mwSize d, mwSize nA, double *R, double *F_A, double thau, 
                                                   double ***pR_to_cells , mwSize *FA_map, mwSize *FA_n){
    mwSize *idx_dF_A,*uF_A;
    mwSize midx_A; 
    if ((d<1)||(nA<1))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error, zero or negative columns or dimension");
    
    idx_dF_A = new mwSize[nA];//(double*)mxCalloc(nA, sizeof(double));
    uF_A     = new mwSize[nA];//(double*)mxCalloc(nA, sizeof(double));
    if (!idx_dF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    
    compute_dist_idx(nA, F_A, &midx_A, idx_dF_A, uF_A ); 

    compute_R_reginv_forall(d, nA, R, F_A, uF_A, idx_dF_A, midx_A, thau, pR_to_cells, FA_map, FA_n);

    delete idx_dF_A;//mxFree(idx_dF_A);
    delete uF_A;
}
void distance_matrix_noindex( mwSize d, mwSize nA, double *A, double *F_A, mwSize nB, double *B, double *F_B, double gamma, 
                                                   double* dm , mwSize *FA_map, mwSize*FB_map  ){
    mwSize *idx_dF_A,*uF_A;
    mwSize midx_A; 
    mwSize *idx_dF_B,*uF_B;
    mwSize midx_B; 
    if ((d<1)||(nA<1)||(nB<1))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error, zero or negative columns or dimension");
    
    idx_dF_A = new mwSize[nA];//(double*)mxCalloc(nA, sizeof(double));
    uF_A     = new mwSize[nA];//(double*)mxCalloc(nA, sizeof(double));
    if (!idx_dF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    
    compute_dist_idx(nA, F_A, &midx_A, idx_dF_A, uF_A ); 
    //print_vec(idx_dF_A,midx_A);
    ////mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","\n");
    
    idx_dF_B = new mwSize[nB];//= (double*)mxCalloc(nB, sizeof(double));
    uF_B     = new mwSize[nB];//(double*)mxCalloc(nA, sizeof(double));
    if (!idx_dF_B)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");

    compute_dist_idx(nB, F_B, &midx_B, idx_dF_B, uF_B); 
    //print_vec(idx_dF_B,midx_B);
    ////mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","\n");
    distance_matrix(d, nA, A, F_A, uF_A, idx_dF_A, midx_A, nB, B, F_B, uF_B, idx_dF_B, midx_B, gamma, dm, FA_map, FB_map  );
    
    delete idx_dF_B;//mxFree(idx_dF_B);
    delete uF_B;
    delete idx_dF_A;//mxFree(idx_dF_A);
    delete uF_A;
}
void distance_matrix_noindex_sameset( mwSize d, mwSize nA, double *A, double *F_A, double gamma, 
                                                   double* dm , mwSize *FA_map){
    mwSize *idx_dF_A,*uF_A;
    mwSize midx_A; 
    idx_dF_A = new mwSize[nA];//= (double*)mxCalloc(1000, sizeof(double));
    uF_A     = new mwSize[nA];//(double*)mxCalloc(nA, sizeof(double));
    if (!idx_dF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    
    compute_dist_idx(nA, F_A, &midx_A, idx_dF_A, uF_A); 
//     print_vec(idx_dF_A,midx_A);
//     mexPrintf("midx_A:%d\n",midx_A);
    distance_matrix_sameset(d, nA, A, F_A, uF_A, idx_dF_A, midx_A, gamma, dm, FA_map);
    
    delete idx_dF_A;//mxFree(idx_dF_A);
    delete uF_A;
}
void rbf_emp_matrix_noindex( mwSize d, mwSize nA, double *A, double *F_A, mwSize nB, double *B, double *F_B, double gamma, 
                                                   double* dm , mwSize *FA_map, mwSize*FB_map  ){
    mwSize *idx_dF_A,*uF_A;
    mwSize midx_A; 
    mwSize *idx_dF_B,*uF_B;
    mwSize midx_B; 
    if ((d<1)||(nA<1)||(nB<1))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error, zero or negative columns or dimension");
    
    
    idx_dF_A = new mwSize[nA];//(double*)mxCalloc(nA, sizeof(double));
    uF_A     = new mwSize[nA];//(double*)mxCalloc(nA, sizeof(double));
    if (!idx_dF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");    
    compute_dist_idx(nA, F_A, &midx_A, idx_dF_A, uF_A ); 
    //print_vec(idx_dF_A,midx_A);
    //mexPrintf("\n");
    
    idx_dF_B = new mwSize[nB];//= (double*)mxCalloc(nB, sizeof(double));
    uF_B     = new mwSize[nB];//(double*)mxCalloc(nA, sizeof(double));
    if (!idx_dF_B)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    compute_dist_idx(nB, F_B, &midx_B, idx_dF_B, uF_B); 
    
    //print_vec(idx_dF_B,midx_B);
    //mexPrintf("\n");
    rbf_emp_matrix(d, nA, A, F_A, uF_A, idx_dF_A, midx_A, nB, B, F_B, uF_B, idx_dF_B, midx_B, gamma, dm, FA_map, FB_map  );
    
    delete idx_dF_B;//mxFree(idx_dF_B);
    delete uF_B;
    delete idx_dF_A;//mxFree(idx_dF_A);
    delete uF_A;
}