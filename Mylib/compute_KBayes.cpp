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
     if(nrhs!=5) {
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
/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double gamma,thau, gamma_r;              /* input scalar */
    double *A,*F_A;
    mwSize *FA_map,*uF_A; 
    mwSize *FA_n;
    size_t nrows, nA, mA, d;                   /* size of matrix */
    double *R,*KB;              /* output matrix */
    double  ***p_to_cells;
    mxArray *cell_mu_R_all;
    mxArray *cell_R_reginv_all;
    mwIndex i;
    mwSize  n_uniq;
    mxArray** pR;
    mxArray** pR_reginv;
    double*  pRd;
    mwSize *idx_dF_A;
    /* check for proper number of arguments */
    validateinputs(nlhs, plhs, nrhs, prhs);

    mA     = mxGetM(prhs[0]);
    nA     = mxGetN(prhs[0]);
    d      = mA;
    gamma  = mxGetScalar(prhs[2]);
    gamma_r= mxGetScalar(prhs[3]);
    thau   = mxGetScalar(prhs[4]);
    A      = mxGetPr(prhs[0]);
    F_A    = mxGetPr(prhs[1]);
    
    n_uniq = get_numberof_dist(nA, F_A);
    
    // get number and indices of each distributions  
    FA_map = new mwSize[n_uniq];
    uF_A   = new mwSize[n_uniq];
    FA_n   = new mwSize[n_uniq];
    idx_dF_A = new mwSize[nA];//(double*)mxCalloc(nA, sizeof(double));
    
    if (!idx_dF_A)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: CAN NOT ALLOCATE MEMORY");
    mwSize midx_A;
    compute_dist_idx(nA, F_A, &midx_A, idx_dF_A, uF_A );
    if (midx_A!=n_uniq)
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","ERORR: Not equal elements and indices in inputs");
    get_FA_n(nA, F_A, uF_A, idx_dF_A, midx_A, FA_n); 
    // allocate a pointer to array of pointers to arrays for R_reginv and mu_R_i for all 
    double*** pResult_to_cells  = (double ***)new double*;
    double*** pR_reginv_to_cells= (double ***)new double*;
    // allocate array of pointers to arrays for R_reginv and mu_R_i
    double **pResult = new double*[n_uniq];
    double **pR_rinv = new double*[n_uniq];
    *pResult_to_cells   = pResult;
    *pR_reginv_to_cells = pR_rinv;
    // alocate actual memory to return to matlab
    pR        = new mxArray*[n_uniq];
    pR_reginv = new mxArray*[n_uniq];
    for( i=0; i<(mwIndex)n_uniq; i++){
        mwSize n_P_i = FA_n[i];
        // for mu_R_i
        pR[i]        = mxCreateDoubleMatrix(n_P_i, 1, mxREAL);
        pResult[i]   = mxGetPr(pR[i]);
        // for R_reginv
        pR_reginv[i] = mxCreateDoubleMatrix(n_P_i, n_P_i, mxREAL);
        pR_rinv[i]   = mxGetPr(pR_reginv[i]);
    }
    /* create the output matrices */
    plhs[0]   = mxCreateDoubleMatrix(nA, nA, mxREAL);
    plhs[1]   = mxCreateDoubleMatrix(n_uniq, n_uniq, mxREAL);
    plhs[2]   = mxCreateCellMatrix((mwSize)n_uniq,1);
    plhs[3]   = mxCreateCellMatrix((mwSize)n_uniq,1);
    /* get a pointer to the real data in the output matrix */
    R         = mxGetPr(plhs[0]);
    KB        = mxGetPr(plhs[1]);
    /* call the computational routine : compute_KBayes to compute KB, R, mu_R_i and R_reginv */
    compute_KBayes(d, nA, A, F_A, uF_A, idx_dF_A, midx_A, gamma, gamma_r, thau, KB, R, pResult_to_cells, pR_reginv_to_cells, FA_map, FA_n);
    // set the cells to the output
    cell_mu_R_all     = plhs[2];
    cell_R_reginv_all = plhs[3];
    for( i=0; i<(mwIndex)n_uniq; i++){
        mxSetCell(cell_mu_R_all,i, pR[i]);
        mxSetCell(cell_R_reginv_all,i, pR_reginv[i]);
    }
    // delete allocated space in mex
    delete pResult_to_cells;
    delete pR_reginv_to_cells;
    delete pResult;
    delete pR_rinv;
    delete pR;
    delete pR_reginv;
    delete idx_dF_A;
    delete FA_n;
    delete uF_A;
    delete FA_map;
}