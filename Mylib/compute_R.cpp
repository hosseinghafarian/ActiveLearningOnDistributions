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

void validate_inputs(mwSize *d, mwSize* nA, int nlhs, mxArray *plhs[],
                                                    int nrhs, const mxArray *prhs[] ) { 
    size_t mA;
    /* check for proper number of arguments */
    if(nrhs!=3) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","Seven inputs required.");
    }
    /* make sure the first input argument is type double */
    if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Input matrix must be type double.");
    }
    mA  = mxGetM(prhs[0]);
    *nA  = mxGetN(prhs[0]);
    
    /* make sure the second input argument is type double */
    if( (!mxIsNumeric(prhs[1]) ) || mxGetN(prhs[1])!= *nA || mxGetM(prhs[1])!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Second argument must be a integer row vector the same size as columns of the first.");
    }
    if( !mxIsNumeric(prhs[2]) && mxIsDouble(prhs[2])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Third argument must be a row vector of integers");
    }
    
    /* make sure the fourth input argument is type double */
    if( !mxIsDouble(prhs[3]) ||  mxIsComplex(prhs[3])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Input matrix must be type double.");
    }
    *d = mA;    
    /* make sure the Fifth input argument is type double */
    if (!mxIsDouble(prhs[3])){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","gamma must be a double");
    }
}
/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double gamma;              /* input scalar */
    double *A, *F_A, *uF_A;               /* 1xN input matrix */

    mwSize nrows, nA, mA, d;                   /* size of matrix */
    double *F_A_map;
    double *outMatrix;              /* output matrix */
   
    validate_inputs(&d,&nA,nlhs,plhs,nrhs,prhs );
     
    /* get the value of the scalar input  */
    gamma = mxGetScalar(prhs[2]);

    /* create a pointer to the real data in the input matrix  */
    A         = mxGetPr(prhs[0]);
    F_A       = mxGetPr(prhs[1]);
    
    /* create the output matrix */
    plhs[0]   = mxCreateDoubleMatrix(nA, nA, mxREAL);
    /* get a pointer to the real data in the output matrix */
    outMatrix = mxGetPr(plhs[0]);
    //distance_matrix(d, nA, A, F_A, midx_A, idx_dF_A, nB, B, F_B, midx_B, idx_dF_B, gamma, outMatrix );
    R_forall_noindex(d, nA, A, F_A, gamma, outMatrix);
}