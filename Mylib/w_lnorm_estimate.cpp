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
    double *F_A, *F_id, *w_l_norm_t, *z_l; 
    mwSize *FA_n;
    mwSize *initL;
    size_t  nA;                   /* size of matrix */
    double  *R,*KB_l, *KB_lu_z_l, *KB_uu;              /* output matrix */
    const mxArray *cell_mu_R_all;
    const mxArray *cell_R_reginv_all;
    mwIndex i;
    mwSize n_uniq, midx_A, n_l;
    mwSize *idx_dF_A;
    mwSize *uF_A;
    /* check for proper number of arguments */
    //validateinputs(nlhs, plhs, nrhs, prhs);
    n_l    = mxGetM(prhs[0]);
    KB_l   = mxGetPr(prhs[0]);
    KB_uu  = mxGetPr(prhs[1]);
    KB_lu_z_l = mxGetPr(prhs[2]);
    R      = mxGetPr(prhs[3]);
    nA     = mxGetN(prhs[3]);
    F_A    = mxGetPr(prhs[4]);
    F_id   = mxGetPr(prhs[5]);
    if ( mxGetM(prhs[1])!= mxGetM(prhs[2]))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: unlabeled instances doesn't match");
    if ( mxGetM(prhs[0])!= mxGetN(prhs[0]))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: First block of kernel matrix must be square");
    if ( mxGetM(prhs[1])!= mxGetN(prhs[1]))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: Second block of kernel matrix must be square");
    if(mxGetN(prhs[2])!=1)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: KB_lu_z_l must be a vector not a matrix");

    n_uniq = get_numberof_dist(nA, F_A);
    
    cell_mu_R_all = prhs[6];
    cell_R_reginv_all = prhs[7];
    initL  = new mwSize[n_l];
    get_int_data(mxGetPr(prhs[8]), n_l, initL);
    //change_indices_tozoeroindexing(initL, n_l); This was a mistake, since indices in initL are distributions ids which are id and not indices
    if( n_l != mxGetM(prhs[9]))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: number of labeled instances don't match");
    z_l    = mxGetPr(prhs[9]);
    // allocate space for number and indices of each distributions  
    FA_n   = new mwSize[n_uniq];
    uF_A   = new mwSize[n_uniq];
    idx_dF_A = new mwSize[nA];
    // get number and indices of each distributions  
    if (!idx_dF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    compute_dist_idx(nA, F_A, &midx_A, idx_dF_A, uF_A );
    if(!isvalid_initL(initL, n_l, uF_A, midx_A))
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: some of initL elements are not in distributions IDs");
    if (midx_A!=n_uniq)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: Not equal elements and indices in inputs");
    get_FA_n(nA, F_A, uF_A, idx_dF_A, midx_A, FA_n); 
    /* create the output matrices */
    plhs[0]   = mxCreateDoubleMatrix(midx_A, 1, mxREAL);
    /* get a pointer to the real data in the output matrix */
    w_l_norm_t= mxGetPr(plhs[0]);
    //print_vec(initL, n_l);
    //print_vec(z_l, n_l);
    /* call the computational routine : compute_KBayes to compute KB, R, mu_R_i and R_reginv */
    get_wnorm_lestimate(KB_l, KB_uu, KB_lu_z_l, R, nA, z_l, initL, n_l, FA_n, cell_mu_R_all, cell_R_reginv_all, uF_A,idx_dF_A, midx_A, w_l_norm_t );
    // delete allocated space 
    delete idx_dF_A;
    delete FA_n;
    delete initL;
}