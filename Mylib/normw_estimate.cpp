/*==========================================================
 * norm_diff_of_distribution.c - 
 *
 * Computes distance between two emprical distrbutions using 
 * Embdedded Kernel  
 * P_1 : A distribution samples in the form d x nA matrix (A)
 * P_2 : A distribution samples in the form d x nB matrix (B)
 * gamma: gamma of RBF kernel for embedding kernel
 * and outputs a 1x1 matrix (outMatrix)
 *
 * The calling syntax is:
 *
 *		outMatrix = norm_diff_of_distrbution(A,B, gamma)
 *
 * This is a MEX-file for MATLAB.
 * Written by Hossein Ghafarian. 
 *
 *========================================================*/
#include "math.h"
#include "mex.h"
#include "kernel_lib.h"
#include "mymatrix_lib.h"
void validateinputs(int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[]){
     if(nrhs!=6) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","Two inputs required.");
    }
//     if(nlhs!=1) {
//         mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","One output required.");
//     }
    /* make sure the first input argument is type double */
    if( !mxIsDouble(prhs[0]) || 
         mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Input matrix must be type double.");
    }
    /* make sure the second input argument is type double */
    if( !mxIsDouble(prhs[1]) || 
         mxIsComplex(prhs[1])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Input matrix must be type double.");
    }
    /* make sure the second input argument is type double */
    if( !mxIsDouble(prhs[2]) || 
         mxIsComplex(prhs[2])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Input matrix must be type double.");
    }
    /* make sure the third input argument is scalar */
    if( !mxIsDouble(prhs[3]) || 
         mxIsComplex(prhs[3]) ||
         mxGetNumberOfElements(prhs[3])!=1 ) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notScalar","gamma must be a scalar.");
    }   
}
void get_int_data(const double *in_prhs, mwSize n_l, mwSize *int_in){
    for(mwSize i=0;i<n_l;i++) 
        int_in[i] = (mwSize) in_prhs[i];
}
int isvalid_initL(mwSize *initL, mwSize n_l, mwSize *uF_A, mwSize n_uFA){
    
    for ( mwSize i=0; i<n_l;i++){
        int found = 0;
        for(mwSize j= 0; j<n_uFA&& !found;j++)
            if(initL[i] == uF_A[j]){
                found = 1;
            }
        if(!found) return 0;
    }
    return 1;
}
void change_indices_tozoeroindexing(mwSize *int_in, mwSize n_l){
    for(mwSize i=0;i<n_l;i++){
        if (int_in[i]>0)
            int_in[i] = int_in[i]-1;
        else
            mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR:zero index in a indexing starting from one.");    
    }
}
/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{   
    double *w_norm_t, *y_l; 
    double *F_A;
    mwSize *FA_n, *F_id;
    mwSize *Lind, *Uind;
    size_t  nA;                   /* size of matrix */
    double  *R,*KB, *KBreg_inv;              /* output matrix */
    const mxArray *cell_mu_R_all;
    const mxArray *cell_R_reginv_all;
    double tol;
    mwIndex i;
    mwSize n_uniq, midx_A, n_l;
    mwSize *idx_dF_A;
    mwSize *uF_A;
    /* check for proper number of arguments */
    //validateinputs(nlhs, plhs, nrhs, prhs);
    n_l       = mxGetM(prhs[0]);
    y_l       = mxGetPr(prhs[0]);
    KB        = mxGetPr(prhs[3]);
    KBreg_inv = mxGetPr(prhs[4]);
    R         = mxGetPr(prhs[5]);
    nA        = mxGetN(prhs[5]);    
    //Lind    = mxGetPr(prhs[1]);
    //Uind    = mxGetPr(prhs[2]);
    if ( mxGetM(prhs[1])!= mxGetM(prhs[2]))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: Lind and Uind must be the same size.");
    if ( mxGetN(prhs[1])!= n_l && mxGetM(prhs[1])!= n_l) 
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: Lind and y_l must be the same size.");
    mwSize midx_ALUF = mxGetN(prhs[7]);
    
    if ( mxGetM(prhs[3])!= mxGetN(prhs[3]))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: First block of kernel matrix must be square");
    
    if ( mxGetM(prhs[4])!= mxGetN(prhs[4]))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: Second block of kernel matrix must be square");
    

    F_A      = mxGetPr(prhs[6]);
    n_uniq = get_numberof_dist(nA, F_A);
//     mexPrintf("n_uniq:%d", n_uniq);
    
    if(mxGetN(prhs[6])!=nA)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: Incosistent number of vectorial instances");
    if(midx_ALUF!=n_uniq)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: Incosistent number of instances");
    
    cell_mu_R_all     = prhs[8];
    cell_R_reginv_all = prhs[9];
    tol               = mxGetScalar(prhs[10]);
    F_id     = new mwSize[n_uniq];
    get_int_data(mxGetPr(prhs[7]), n_uniq, F_id);
    
    
    // allocate space for number and indices of each distributions  
    FA_n     = new mwSize[n_uniq];
    if (!FA_n)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    uF_A     = new mwSize[n_uniq];
    if (!uF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    idx_dF_A = new mwSize[nA];
    // get number and indices of each distributions  
    if (!idx_dF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    compute_dist_idx(nA, F_A, &midx_A, idx_dF_A, uF_A );
// //     if(!isvalid_initL(initL, n_l, uF_A, midx_A))
// //         mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: some of initL elements are not in distributions IDs");
    if (midx_A!=n_uniq)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: Not equal elements and indices in inputs");
    mwSize n_u = midx_A - n_l;
    Lind     = new mwSize[n_l];   
    get_int_data(mxGetPr(prhs[1]), n_l, Lind);
    for(mwSize j=0; j<n_l;j++) Lind[j] -= 1;
    
    Uind     = new mwSize[n_u];
    get_int_data(mxGetPr(prhs[2]), n_u, Uind);
    for(mwSize j=0; j<n_u;j++) Uind[j] -= 1;
    
    get_FA_n(nA, F_A, uF_A, idx_dF_A, midx_A, FA_n); 
    /* create the output matrices */
    plhs[0]   = mxCreateDoubleMatrix(midx_A, 1, mxREAL);
   /* get a pointer to the real data in the output matrix */
    w_norm_t  = mxGetPr(plhs[0]);
    //w_norm_t[0] = 1.0;
    //print_vec(initL, n_l);
    //print_vec(z_l, n_l);
    /* call the computational routine : compute_KBayes to compute KB, R, mu_R_i and R_reginv */
    get_wnorm_estimate(y_l, Lind, Uind, KB, KBreg_inv, R, FA_n, F_id, cell_mu_R_all, cell_R_reginv_all,
                        nA, n_l, uF_A, idx_dF_A, midx_A, w_norm_t , tol);
    // delete allocated space 
    delete F_id;
    delete idx_dF_A;
    delete uF_A;
    delete FA_n;    
    delete Lind;
    delete Uind;
}