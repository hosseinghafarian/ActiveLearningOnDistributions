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
    double gamma,thau;              /* input scalar */
    double *A,*F_A;
    mwSize *FA_map; 
    mwSize *FA_n;
    size_t nrows, nA, mA, d;                   /* size of matrix */
    double    *R;              /* output matrix */
    double  ***p_to_cells;
    mwSize n_uniq;
    /* check for proper number of arguments */
    validateinputs(nlhs, plhs, nrhs, prhs);

    mA     = mxGetM(prhs[0]);
    nA     = mxGetN(prhs[0]);
    d      = mA;
    gamma  = mxGetScalar(prhs[2]);
    thau   = mxGetScalar(prhs[3]);
    A      = mxGetPr(prhs[0]);
    F_A    = mxGetPr(prhs[1]);

    n_uniq = get_numberof_dist(nA, F_A);

    FA_map = new mwSize[n_uniq];
    FA_n   = new mwSize[n_uniq];
    p_to_cells = (double ***)new double*;
    /* create the output matrix */
    plhs[0]   = mxCreateDoubleMatrix(nA, nA, mxREAL);
    /* get a pointer to the real data in the output matrix */
    R         = mxGetPr(plhs[0]);
    /* call the computational routine */
    R_forall_noindex(d, nA, A, F_A, gamma, R);
    compute_R_reginv_forall_noindex( d, nA, R, F_A, thau, p_to_cells, FA_map, FA_n);
    // delete allocated memory
    delete_p_to_cells(n_uniq, p_to_cells);
    //delete p_to_cells;
    delete FA_n;
    delete FA_map;
}