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
            mexPrintf(",%7.5f",p[i+j*n]);
        mexPrintf(";");
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
    ////mexPrintf("\n norm in RBF is:%10.7f gamma:%10.7f",sum,gamma);
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
            ////mexPrintf("\n");print_vec(&A[i*d],d);//mexPrintf(" with vec"); print_vec(&A[j*d],d);//mexPrintf("val:%f\n", value);
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
/* This function computes the value of meanembeding of distribution A at each location of instances drawn from distribution itself, 
 * it's a real array in the same dimension as the number of instances drawn from distribution*/
void meanembed_fval_at_all(double gamma, mwSize d, mwSize nA, double *A, double* vec_out){
     mwSize i;
     for(i=0; i< nA; i++ ) 
         vec_out[i] = meanembed_funcval_at_x(gamma, d, nA, A, &A[i*d]);
}
/* This function computes meanembeding of distributions at each point of each distribution. 
 * It's output is a pointer to an array of pointers to array of embedding of distrributions for each point*/
void meanembed_fval_dist( mwSize d, mwSize nA, double *A, double *F_A, double*uF_A, mwSize midx_A, double *idx_dF_A, double gamma, 
                                double ***p_to_cells, double* FA_map, mwSize *FA_n){
    const double tol = 1e-12;
    double *X_i;
    double **p;

    mwSize i, stidx_A, enidx_A, nX_i, nX_j;
   
    p  = new double*[midx_A];
    *p_to_cells = p;
    
    stidx_A = 0;
    for(i=0; i < midx_A;i++){
        enidx_A = idx_dF_A[i];
       if (enidx_A <0 ){
              mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error negative index in idx_dF_B");
       }
       FA_map[i] = uF_A[i]; 
       X_i       = &A[stidx_A*d]; nX_i = enidx_A-stidx_A;
       p[i]      = new double[nX_i];
       FA_n[i]   = nX_i;
       meanembed_fval_at_all(gamma, d, nX_i, X_i, p[i]);
       stidx_A   = enidx_A;
    }    
}
// this function deletes an structure in the form of pointer to an array of pointers to arrays         
void delete_p_to_cells(mwSize midx_A, double ***p_to_cells, mwSize *FA_n){
    mwSize i, nX_i;
    for(i=0; i < midx_A;i++){
       double *p = (*p_to_cells)[i];
       delete (*p_to_cells)[i];
    }
    delete p_to_cells;
}
/* This is the same as function meanembed_fval_dist but without index of distributions*/ 
void meanembed_noindex_dist( mwSize d, mwSize nA, double *A, double *F_A, double* uF_A, double gamma, 
                                                   double ***p_to_cells , double *FA_map, mwSize *FA_n){
    double *idx_dF_A;
    mwSize midx_A; 
    if ((d<1)||(nA<1))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error, zero or negative columns or dimension");
    
    idx_dF_A = new double[nA];//(double*)mxCalloc(nA, sizeof(double));
    if (!idx_dF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    
    compute_dist_idx(nA, F_A, &midx_A, idx_dF_A ); 
    meanembed_fval_dist(d, nA, A, F_A, uF_A, midx_A, idx_dF_A, gamma, p_to_cells, FA_map, FA_n);
    delete idx_dF_A;
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
void rbf_emp_matrix( mwSize d, mwSize nA, double *A, double *F_A, double*uF_A, mwSize midx_A, double *idx_dF_A, 
                                mwSize nB, double *B, double *F_B, double*uF_B, mwSize midx_B, double*idx_dF_B, double gamma, 
                                double* dm, double *FA_map, double*FB_map  ){
    const double tol = 1e-12;
    double *X_i, *X_j,value;
    mwSize i,j, stidx_A, stidx_B, enidx_A, enidx_B, nX_i, nX_j;
    stidx_A = 0;
    for(i=0; i < midx_A;i++){
       enidx_A = idx_dF_A[i];
       if (enidx_A <0 ){
              mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error negative index in idx_dF_B");
       }
       //mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","enidx_A:%f\n",enidx_A);
       FA_map[i] = uF_A[i]; 
       X_i = &A[stidx_A*d]; nX_i = enidx_A-stidx_A;
       stidx_B = 0;
       for(j =0; j<midx_B;j++){
           enidx_B = idx_dF_B[j];
           if (enidx_B <0){ 
              mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error negative index in idx_dF_B");
           }
           FB_map[j] = uF_B[j];
           X_j = &B[stidx_B*d]; nX_j = enidx_B-stidx_B;
           if (nX_j <0) 
               mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error negative lenght");
           
           value = rbf_emp_kernel(gamma, d, nX_i, nX_j, X_i, X_j);
           ////mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","\n");print_matrix(X_i,d,nX_i);//mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","with");print_matrix(X_j,d,nX_j); //mexPrintf(" gamma:%f, value:%10.8f",gamma, i,j,value);
           if (value <tol){
               value = 0;
           }
           //dm[i*midx_A+j] = value;
           dm[j*midx_A+i] = value;
           stidx_B = enidx_B;
       }
       stidx_A = enidx_A;
    }
    ////mexPrintf("end of distance_matrix_reached\n");
}
/* This function computes the R kernel matrix which is a kernel such that the draws from the corresponding 
 * distribution generate distributions which are in embedding kernel space, see paper: Bayesian learning of kernel mean embedding
 * It is a matrix for kernels between any instance in any distributions with any other instance*/
void R_forall_indexed( mwSize d, mwSize nA, double *A, double *F_A, double*uF_A, mwSize midx_A, double *idx_dF_A, double gamma_r, 
                                double* R){
    const double tol = 1e-12;
    double *X_i, *X_j,value;
    mwSize i,j, stidx_A, stidx_B, enidx_A, enidx_B, nX_i, nX_j;
    stidx_A = 0;
    for(i=0; i < nA; i++){
       X_i = &A[i*d];
       for(j =0; j<=i;j++){
           
           X_j = &A[j*d];
           value = rbf_kernel_compute(gamma_r, d, X_i, X_j);
           if (value <tol){
               value = 0;
           }
           //mexPrintf("R(%d, %d):%f\n", i,j,value);
           R[j*nA+i] = value;
           R[i*nA+j] = value;
       }
    }
}
/* This is the above function except that it doesnot have indices of distributions*/
void R_forall_noindex( mwSize d, mwSize nA, double *A, double *F_A, double* uF_A, double gamma_r, 
                                                   double* R){
    double *idx_dF_A;
    mwSize midx_A; 

    if ((d<1)||(nA<1))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error, zero or negative columns or dimension");
    
    idx_dF_A = new double[nA];//(double*)mxCalloc(nA, sizeof(double));
    if (!idx_dF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    
    compute_dist_idx(nA, F_A, &midx_A, idx_dF_A ); 

    R_forall_indexed(d, nA, A, F_A, uF_A, midx_A, idx_dF_A, gamma_r, R);
    
    delete idx_dF_A;//mxFree(idx_dF_A);
}
/* This function copies an square submatrix of Matrix R from stidx_A to enidx_A*/
void copy_matrix_part(mwSize nA, mwSize stidx_A, mwSize enidx_A, double*R, double* R_reginv_i){
    mwSize nX_i= enidx_A-stidx_A;
    for(mwSize i= 0;i<nX_i;i++)
        for(mwSize j= 0;j<nX_i;j++)
            R_reginv_i[j*nX_i+i ] = R[(j+stidx_A)*nA+(i+stidx_A)];
}
/* This function adds a regularization term to maindiagonal of matrix R_i*/
void reg_matrix(mwSize nR_i, double* R_i, double regterm){
    for(mwSize j= 0;j<nR_i;j++)
        R_i[j*nR_i+j ] = R_i[j*nR_i+j ] + regterm;
}
/* This function extract a submatrix of regarding a distribution from R and then regualrize it using the above function and finaly 
  Inverse the matrix */
void R_reginv_dist(mwSize nA, mwSize stidx_A, mwSize enidx_A, double*R, double* R_reginv_i, double thau){
    //mexPrintf(" stidx_A:%d, enidx_A:%d", stidx_A, enidx_A);
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
void compute_R_reginv_forall( mwSize d, mwSize nA, double *R, double *F_A, double*uF_A, mwSize midx_A, double *idx_dF_A, double thau, 
                                double ***pR_to_cells, double* FA_map, mwSize *FA_n){
    const double tol = 1e-12;
    double *X_i;
    double **p;
    //print_vec(idx_dF_A, midx_A);
    mwSize i, stidx_A, enidx_A, nX_i, nX_j;
   
    p = *pR_to_cells;
    
    stidx_A = 0;
    
    for(i=0; i < midx_A;i++){
       enidx_A = idx_dF_A[i];
       if (enidx_A <0 ){
              mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error negative index in idx_dF_B");
       }
       FA_map[i] = uF_A[i]; 
       nX_i      = enidx_A-stidx_A;
       FA_n[i]   = nX_i;
       R_reginv_dist(nA, stidx_A, enidx_A, R, p[i], thau);
       stidx_A   = enidx_A;
    }    
}
void get_FA_n(mwSize nA, double *F_A, double*uF_A, mwSize midx_A, double *idx_dF_A, 
                                mwSize *FA_n){
    mwSize i, stidx_A, enidx_A, nX_i;
   
    stidx_A = 0;
    for(i=0; i < midx_A;i++){
       enidx_A = idx_dF_A[i];
       if (enidx_A <0 ){
              mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error negative index in idx_dF_B");
       }
       nX_i      = enidx_A-stidx_A;
       if (nX_i <=0 ){
              mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Start index less than or equal to end index");
       }
       FA_n[i]   = nX_i;
       stidx_A   = enidx_A;
    }    
}
/* This is the same function as compute_R_reginv_forall but without indices*/ 
void compute_R_reginv_forall_noindex( mwSize d, mwSize nA, double *R, double *F_A, double* uF_A, double thau, 
                                                   double ***pR_to_cells , double *FA_map, mwSize *FA_n){
    double *idx_dF_A;
    mwSize midx_A; 
    if ((d<1)||(nA<1))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error, zero or negative columns or dimension");
    
    idx_dF_A = new double[nA];//(double*)mxCalloc(nA, sizeof(double));
    if (!idx_dF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    
    compute_dist_idx(nA, F_A, &midx_A, idx_dF_A ); 
    compute_R_reginv_forall(d, nA, R, F_A, uF_A, midx_A, idx_dF_A, thau, pR_to_cells, FA_map, FA_n);
    delete idx_dF_A;
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
//        mexPrintf("\n");
//        print_vec(p[i], nX_i);
    }
}
/* This function computes the Kernel matrix between distributions i and j */
double compute_KBayes_ij(mwSize nA, double *R, mwSize  stidx_A_i, mwSize enidx_A_i, mwSize  stidx_A_j, mwSize enidx_A_j,
                                    double* pResult_i, mwSize n_i, double* pResult_j, mwSize n_j ){
    
    double result;// the initial value must be changed 
    
    //mexPrintf("P_i: (%d, %d) P_j: (%d, %d)\n",  stidx_A_i, enidx_A_i, stidx_A_j, enidx_A_j);
    mwSize leadingdim_A = nA;
    double *pR_start = R + stidx_A_j*nA + stidx_A_i;// check if it is correct?
    mwSize dimsubA_1 = enidx_A_i-stidx_A_i;
    mwSize dimsubA_2 = enidx_A_j-stidx_A_j;
    // compute R_ij*pResj = R_ij*(R_j+tha/n_j I)^{-1} mu_P_X_j
    double *pRijResj = new double[dimsubA_1];
    submatrix_multiply(dimsubA_1, dimsubA_2, n_j, 1, pR_start, leadingdim_A, pResult_j, pRijResj) ;
    matrix_multiply(1, dimsubA_1, dimsubA_1, 1,  pResult_i, pRijResj, &result);
    delete pRijResj;
    
    return result;
}
void comp_mu_R_all( mwSize d, mwSize nA, double *A, double *F_A, double* uF_A, double *idx_dF_A, mwSize midx_A, double gamma, double thau, double* R,
                                                   double *** pResult_to_cells, double*** pR_reginv_to_cells, double *FA_map, mwSize *FA_n){
    // compute the Rregiv for all distributions
    
    compute_R_reginv_forall(d, nA, R, F_A, uF_A, midx_A, idx_dF_A, thau, pR_reginv_to_cells, FA_map, FA_n);
    double **pR = *pR_reginv_to_cells;
    // compute the mu_p_X_i for all distributions. 
    double*** p_to_cells = (double ***)new double*; 
    meanembed_noindex_dist( d, nA, A, F_A, uF_A, gamma, p_to_cells, FA_map, FA_n);
    double **pMu_P_X = *p_to_cells;
    // compute mu_R_i= Rregiv_i*mu_p_X_i = (R_i + thau/n_i)^{-1} mu_p_X_i;
    multiply_mu_P_Rreginv(pR_reginv_to_cells, p_to_cells, pResult_to_cells, midx_A, FA_n);

    delete_p_to_cells(midx_A, p_to_cells, FA_n);
}
/* This function computes the matrix KBayes between each distributions and also, 
 * computes and returns matrix R*/
void compute_KBayes( mwSize d, mwSize nA, double *A, double *F_A, double* uF_A, double* idx_dF_A, mwSize midx_A, double gamma, double gamma_r, double thau, 
                                                  double *KB, double* R, double *** pResult_to_cells, double*** pR_reginv_to_cells, double *FA_map, mwSize *FA_n){
      
    if ((d<1)||(nA<1))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error, zero or negative columns or dimension");
    
    if (!idx_dF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: indices of distributions didnot computed");
    // compute the R matrix 
    R_forall_indexed(d, nA, A, F_A, uF_A, midx_A, idx_dF_A, gamma_r, R);
    // compute mu_R_i= Rregiv_i*mu_p_X_i = (R_i + thau/n_i)^{-1} mu_p_X_i;
    
    comp_mu_R_all( d, nA, A, F_A, uF_A, idx_dF_A, midx_A, gamma, thau, R, pResult_to_cells, pR_reginv_to_cells, FA_map, FA_n);
    double **pResult = *pResult_to_cells;
    // compute KBayes
    double val;
    mwSize enidx_A_i, enidx_A_j;
    mwSize stidx_A_i = 0;
    for(mwSize i=0; i < midx_A;i++){
       enidx_A_i = idx_dF_A[i];        
       mwSize stidx_A_j = 0;
       mwSize n_i = FA_n[i];
       for(mwSize j =0; j<=i ;j++){
           enidx_A_j = idx_dF_A[j];
           mwSize n_j = FA_n[j];
           if (n_j >= n_i){ // first multiply larger dimension to make a smaller vector.
               val = compute_KBayes_ij(nA, R, stidx_A_i, enidx_A_i, stidx_A_j, enidx_A_j,
                                                 pResult[i], n_i, pResult[j], n_j);
           }
           else{
               val = compute_KBayes_ij(nA, R, stidx_A_j, enidx_A_j, stidx_A_i, enidx_A_i,
                                                 pResult[j], n_j, pResult[i], n_i);
           }
           KB[j*midx_A+i] = val;
           KB[i*midx_A+j] = val;
           stidx_A_j = enidx_A_j;
       }
       stidx_A_i = enidx_A_i;
    }
}
void compute_KBayes_noindex( mwSize d, mwSize nA, double *A, double *F_A, double* uF_A, double gamma, double gamma_r, double thau, 
                                                  double *KB, double* R, double *** pResult_to_cells, double*** pR_reginv_to_cells, double *FA_map, mwSize *FA_n){  
    double *idx_dF_A;
    mwSize midx_A; 
    if ((d<1)||(nA<1))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error, zero or negative columns or dimension");
    
    idx_dF_A = new double[nA];//(double*)mxCalloc(nA, sizeof(double));
    if (!idx_dF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    
    compute_dist_idx(nA, F_A, &midx_A, idx_dF_A );
    // compute the R matrix 
    compute_KBayes(d, nA, A, F_A, uF_A, idx_dF_A, midx_A, gamma, gamma_r, thau, KB, R, pResult_to_cells, pR_reginv_to_cells,  FA_map, FA_n);
    
    delete idx_dF_A;
}
int  foundin(mwSize item, mwSize* initL, mwSize n_l){
     
     for(mwSize j=0; j<n_l;j++)
        if( initL[j]==item)
           return 1;
     mexPrintf("not found item:%d in :\n");
     print_vec(initL, n_l);
     return 0;
}
// This function computes nu_t = \sum_{i\in D_l} R^{ti} *mu_R_i*z_l[i]
void comp_nu_t(double*R, const mxArray* mu_R_icell, mwSize stidx_A_t, mwSize enidx_A_t, mwSize nA, mwSize midx_A, double* uF_A, double *idx_dF_A, mwSize n_P_t,
               double*z_l, mwSize initL[], mwSize n_l, double* nu_t, mwSize t){
    
    for(mwSize i=0; i<n_P_t; i++) nu_t[i] = 0;
   
    mwSize i, stidx_A_i, enidx_A_i;
    double beta  = 1.0;
    stidx_A_i = 0;
    for(i=0; i < midx_A;i++){
       enidx_A_i = idx_dF_A[i];
       if (enidx_A_i <0 ){
              mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error negative index in idx_dF_B");
       }
       mwSize dist_i = uF_A[i];
        if (foundin(dist_i, initL, n_l)){
            // add R^{ti} *mu_R_i*z_l[i] to nu_t to make nu_t = \sum_{i\in D_l} R^{ti} *mu_R_i*z_l[i]
           const   mxArray* cell_el_ptr = mxGetCell(mu_R_icell, i);
           double* mu_R_i   = mxGetPr(cell_el_ptr);
           mwSize  n_mu_R_i = mxGetM(cell_el_ptr);           
           mexPrintf("\n");
           double alpha = z_l[i];
           mexPrintf("stidx_A_t: %d, enidx_A_t:%d\n",stidx_A_t, enidx_A_t);
           submat_mult_alphbeta(stidx_A_t, enidx_A_t, stidx_A_i, enidx_A_i, nA, nA, alpha, R, 
                                n_mu_R_i, 1, mu_R_i, beta, nu_t);
           print_vec(nu_t, enidx_A_t-stidx_A_t);
       }
       stidx_A_i = enidx_A_i;  
    }
}
void get_wnorm_lestimate(double *KB_l, double *R, mwSize nA, double *z_l, mwSize initL[], mwSize n_l, mwSize* FA_n, 
                         const mxArray* mu_R_icell, const mxArray* R_reginv_cell, mwSize midx_A, double *uF_A, double *idx_dF_A, double* w_l_norm_t ){
    double **nu_array= new double*[midx_A];
    
    mwSize stidx_A_t, enidx_A_t;
    stidx_A_t = 0;
    for(mwSize t=0; t<midx_A; t++){
       enidx_A_t  = idx_dF_A[t]; 
       if (enidx_A_t <0 ){
              mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error negative index in idx_dF_B");
       }
       mwSize n_P_t = FA_n[t];
       nu_array[t] = new double[n_P_t];
       comp_nu_t(R, mu_R_icell, stidx_A_t, enidx_A_t, nA, midx_A, uF_A, idx_dF_A, n_P_t, z_l, initL, n_l, nu_array[t], t);
       stidx_A_t = enidx_A_t;
    }
    double * norm_nu = new double[midx_A];
    //comp_norm_nu_Rreginv(midx_A, nuarray, R_reginv_cell, norm_nu); 
    double z_lKB_lz_l = 1;//comp_z_lnormKB(KB_l, z_l, n_l);
    for(mwSize t=0; t<midx_A; t++){
       w_l_norm_t[t] = z_lKB_lz_l - 0.1;//norm_nu[t];
    }
    delete norm_nu;
    for(mwSize t=0; t<midx_A; t++){
       delete nu_array[t];
    }
    delete nu_array;
}
void distance_matrix( mwSize d, mwSize nA, double *A, double *F_A, double*uF_A, mwSize midx_A, double *idx_dF_A, 
                                mwSize nB, double *B, double *F_B, double*uF_B, mwSize midx_B, double*idx_dF_B, double gamma, 
                                double* dm, double *FA_map, double*FB_map  ){
    const double tol = 1e-12;
    double *X_i, *X_j,value;
    mwSize i,j, stidx_A, stidx_B, enidx_A, enidx_B, nX_i, nX_j;
    stidx_A = 0;
    for(i=0; i < midx_A;i++){
       enidx_A = idx_dF_A[i];
       if (enidx_A <0 ){
              mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error negative index in idx_dF_B");
       }
       //mexPrintf("enidx_A:%f\n",enidx_A);
       FA_map[i] = uF_A[i]; 
       X_i = &A[stidx_A*d]; nX_i = enidx_A-stidx_A;
       stidx_B = 0;
       for(j =0; j<midx_B;j++){
           enidx_B = idx_dF_B[j];
           if (enidx_B <0){ 
              mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error negative index in idx_dF_B");
           }
           FB_map[j] = uF_B[j];
           X_j = &B[stidx_B*d]; nX_j = enidx_B-stidx_B;
           if (nX_j <0) 
               mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error negative lenght");
           
           value = normdiff_of_dist(gamma, d, nX_i, nX_j, X_i, X_j);
           ////mexPrintf("\n");print_matrix(X_i,d,nX_i);//mexPrintf("with");print_matrix(X_j,d,nX_j); //mexPrintf(" gamma:%f, value:%10.8f",gamma, i,j,value);
           if (value <tol){
               value = 0;
           }
           //dm[i*midx_A+j] = value;
           dm[j*midx_A+i] = value;
           stidx_B = enidx_B;
       }
       stidx_A = enidx_A;
    }
    ////mexPrintf("end of distance_matrix_reached\n");
}
// VERY IMPORTANT: ORDER OF uF_A AND F_A MUST BE THE SAME ,i.e. DISTRIBUTION uF_A[i] MUST BE THE i-th DISTINCT DISTRIBUTION IN F_A 
// FROM stidx_A = idx_dF_A[i-1] TO enidx_A = idx_dF_A[i]. IN ORDER TO ENFORCE THIS, WE MUST CHECK THIS IN compute_dist_idx. 
// It is better to compute uF_A based on the F_A, so there is no need to get it from the inputs which is a redundancy. 
void distance_matrix_sameset( mwSize d, mwSize nA, double *A, double *F_A, double* uF_A, mwSize midx_A, double *idx_dF_A, 
                              double gamma, double* dm, double *FA_map){
    const double tol = 1e-12;
    double *X_i, *X_j,value;
    mwSize i,j, stidx_A, stidx_B, enidx_A, enidx_B, nX_i, nX_j;
    stidx_A = 0;
    for(i=0; i < midx_A;i++){
       enidx_A = idx_dF_A[i];
       if (enidx_A <0 ){
              mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error negative index in idx_dF_B");
       }
       FA_map[i] = uF_A[i];
       X_i = &A[stidx_A*d]; nX_i = enidx_A-stidx_A;
       dm[i*midx_A+i] = 0;
       stidx_B = idx_dF_A[i];
       for(j =i+1; j<midx_A;j++){
           enidx_B = idx_dF_A[j];
           X_j = &A[stidx_B*d]; nX_j = enidx_B-stidx_B;
           if (nX_j <0){ 
               mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error negative lenght");
           }
           value = normdiff_of_dist(gamma, d, nX_i, nX_j, X_i, X_j);
           if (value <tol){
               value = 0;
           }
           dm[i*midx_A+j] = value;
           dm[j*midx_A+i] = value;
           stidx_B = enidx_B;
       }
       stidx_A = enidx_A;
    }
}
void mydiff(mwSize nF, double* F, double *dF){
    mwSize i;
    for(i=0;i<nF-1;i++)
        dF[i] = F[i+1]- F[i];
    return;
}
void findnz(mwSize ndF, double* dF, mwSize *max_ndF, double *idx_dF){
    mwSize i,j,m;
    *max_ndF = 0;
    j=0;
    for(i=0;i<ndF;i++)
        if (dF[i]){
            idx_dF[j] = i+1; j++; *max_ndF = j;
        }
}
// VERY IMPORTANT: ORDER OF uF_A AND F_A MUST BE THE SAME ,i.e. DISTRIBUTION uF_A[i] MUST BE THE i-th DISTRIBUTION IN F_A 
// FROM stidx_A = idx_dF_A[i-1] TO enidx_A = idx_dF_A[i]. IN ORDER TO ENFORCE THIS, WE MUST CHECK THIS IN compute_dist_idx. 
void compute_dist_idx(mwSize nA, double* F_A, mwSize* max_ndF, double *idxF ){
    double *tmp_idxF = new double[nA];
    if (!tmp_idxF)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    //double *uF       = new double[nA];
    mydiff(nA, F_A, tmp_idxF);
    findnz(nA-1,tmp_idxF,max_ndF,idxF);
    if ((*max_ndF)> nA-1)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Too many nonzeros");
    idxF[*max_ndF]=nA;
    (*max_ndF)++;
    delete tmp_idxF;
}
void distance_matrix_noindex( mwSize d, mwSize nA, double *A, double *F_A, double* uF_A, mwSize nB, double *B, double *F_B, double* uF_B, double gamma, 
                                                   double* dm , double *FA_map, double*FB_map  ){
    double *idx_dF_A;
    mwSize midx_A; 
    double *idx_dF_B;
    mwSize midx_B; 
    if ((d<1)||(nA<1)||(nB<1))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error, zero or negative columns or dimension");
    
    idx_dF_A = new double[nA];//(double*)mxCalloc(nA, sizeof(double));
    if (!idx_dF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    
    compute_dist_idx(nA, F_A, &midx_A, idx_dF_A ); 
    //print_vec(idx_dF_A,midx_A);
    ////mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","\n");
    
    idx_dF_B = new double[nB];//= (double*)mxCalloc(nB, sizeof(double));
    if (!idx_dF_B)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");

    compute_dist_idx(nB, F_B, &midx_B, idx_dF_B ); 
    //print_vec(idx_dF_B,midx_B);
    ////mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","\n");
    distance_matrix(d, nA, A, F_A, uF_A, midx_A, idx_dF_A, nB, B, F_B, uF_B, midx_B, idx_dF_B, gamma, dm, FA_map, FB_map  );
    
    delete idx_dF_B;//mxFree(idx_dF_B);
    delete idx_dF_A;//mxFree(idx_dF_A);
}
void distance_matrix_noindex_sameset( mwSize d, mwSize nA, double *A, double *F_A, double* uF_A, double gamma, 
                                                   double* dm , double *FA_map){
    double *idx_dF_A;
    mwSize midx_A; 
    idx_dF_A = new double[nA];//= (double*)mxCalloc(1000, sizeof(double));
    if (!idx_dF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");

    compute_dist_idx(nA, F_A, &midx_A, idx_dF_A ); 
    //print_vec(idx_dF_A,midx_A);
    distance_matrix_sameset(d, nA, A, F_A, uF_A, midx_A, idx_dF_A, gamma, dm, FA_map);
    
    delete idx_dF_A;//mxFree(idx_dF_A);
}
void rbf_emp_matrix_noindex( mwSize d, mwSize nA, double *A, double *F_A, double* uF_A, mwSize nB, double *B, double *F_B, double* uF_B, double gamma, 
                                                   double* dm , double *FA_map, double*FB_map  ){
    double *idx_dF_A;
    mwSize midx_A; 
    double *idx_dF_B;
    mwSize midx_B; 
    if ((d<1)||(nA<1)||(nB<1))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Error, zero or negative columns or dimension");
    
    idx_dF_A = new double[nA];//(double*)mxCalloc(nA, sizeof(double));
    if (!idx_dF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    
    compute_dist_idx(nA, F_A, &midx_A, idx_dF_A ); 
    //print_vec(idx_dF_A,midx_A);
    ////mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","\n");
    
    idx_dF_B = new double[nB];//= (double*)mxCalloc(nB, sizeof(double));
    if (!idx_dF_B)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");

    compute_dist_idx(nB, F_B, &midx_B, idx_dF_B ); 
    //print_vec(idx_dF_B,midx_B);
    ////mexPrintf("\n");
    rbf_emp_matrix(d, nA, A, F_A, uF_A, midx_A, idx_dF_A, nB, B, F_B, uF_B, midx_B, idx_dF_B, gamma, dm, FA_map, FB_map  );
    
    delete idx_dF_B;//mxFree(idx_dF_B);
    delete idx_dF_A;//mxFree(idx_dF_A);
}