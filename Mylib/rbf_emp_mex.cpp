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

void validate_inputs(mwSize *d, mwSize* nA, mwSize*nB, int nlhs, mxArray *plhs[],
                                                    int nrhs, const mxArray *prhs[] ) { 
    size_t mA, mB;
    /* check for proper number of arguments */
    if(nrhs!=5) {
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
    /* make sure the fourth input argument is type double */
    if( !mxIsDouble(prhs[2]) ||  mxIsComplex(prhs[2])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Input matrix must be type double.");
    }
    mB  = mxGetM(prhs[2]);
    *nB  = mxGetN(prhs[2]);
    /*check that both of the first and second argument are in the same size*/
    if(mA != mB) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector","First and Fourth Input must have the same dimension.");
    }
    *d = mA;
    
    /* make sure the Fifth input argument is type double */
    if( (!mxIsNumeric(prhs[3]) && mxIsDouble(prhs[3])) || mxGetN(prhs[3])!= *nB || mxGetM(prhs[3])!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Fifth argument must be a integer row vector the same size as columns of the first.");
    }
    if (!mxIsDouble(prhs[4])){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","gamma must be a double");
    }
}
/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double gamma;              /* input scalar */
    double *A, *F_A;               /* 1xN input matrix */
    double *B, *F_B;               /* 1xN input matrix */
    mwSize nrows, nA, mA, nB, mB, d, muF_A, muF_B;                   /* size of matrix */
    mwSize *F_A_map,*F_B_map;
    double *outMatrix;              /* output matrix */
   
    validate_inputs(&d,&nA,&nB,nlhs,plhs,nrhs,prhs );
    mexPrintf("nA:%d", nA);
    /* get the value of the scalar input  */
    gamma = mxGetScalar(prhs[4]);

    /* create a pointer to the real data in the input matrix  */
    A         = mxGetPr(prhs[0]);
    F_A       = mxGetPr(prhs[1]);
    B         = mxGetPr(prhs[2]);
    F_B       = mxGetPr(prhs[3]);
    muF_A     = get_numberof_dist(nA, F_A);
    muF_B     = get_numberof_dist(nB, F_B);
    /* create the output matrix */
    plhs[0]   = mxCreateDoubleMatrix(muF_A, muF_B, mxREAL);
    
    plhs[1]   = mxCreateNumericMatrix(1, muF_A, mxINT64_CLASS, mxREAL);
    F_A_map   =(mwSize*) mxGetData(plhs[1]);
    
    plhs[2]   = mxCreateNumericMatrix(1, muF_B, mxINT64_CLASS, mxREAL);
    F_B_map   =(mwSize*) mxGetData(plhs[2]);
    /* get a pointer to the real data in the output matrix */
    outMatrix = mxGetPr(plhs[0]);
    //distance_matrix(d, nA, A, F_A, midx_A, idx_dF_A, nB, B, F_B, midx_B, idx_dF_B, gamma, outMatrix );
    rbf_emp_matrix_noindex(d, nA, A, F_A, nB, B, F_B, gamma, outMatrix, F_A_map, F_B_map );
}