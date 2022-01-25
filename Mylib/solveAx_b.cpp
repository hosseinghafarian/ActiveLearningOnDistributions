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

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    
    double *A, *b, *x; 
    mwSize nA;

    const mxArray *mA;
    nA        = mxGetM(prhs[0]);
    A         = mxGetPr(prhs[0]);
    b         = mxGetPr(prhs[1]);
    /* create the output matrices */
    plhs[0]   = mxCreateDoubleMatrix(nA, 1, mxREAL);
    /* get a pointer to the real data in the output matrix */
    x         = mxGetPr(plhs[0]);
    //print_vec(initL, n_l);
    //print_vec(z_l, n_l);
    /* call the computational routine : compute_KBayes to compute KB, R, mu_R_i and R_reginv */
    solve_linear(nA, A, b);
    
    for(mwSize i=0; i<nA; i++) x[i] = b[i];
}