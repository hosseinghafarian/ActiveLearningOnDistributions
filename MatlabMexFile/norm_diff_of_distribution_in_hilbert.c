/*==========================================================
 * arrayProduct.c - example in MATLAB External Interfaces
 *
 * Multiplies an input scalar (multiplier) 
 * times a 1xN matrix (inMatrix)
 * and outputs a 1xN matrix (outMatrix)
 *
 * The calling syntax is:
 *
 *		outMatrix = arrayProduct(multiplier, inMatrix)
 *
 * This is a MEX-file for MATLAB.
 * Copyright 2007-2012 The MathWorks, Inc.
 *
 *========================================================*/
#include "math.h"
#include "mex.h"

/* The computational routine */
double rbf_kernel_compute(double gamma, mwSize d, double *y, double *z)
{
    mwSize i;
    double sum, retval;
    /* multiply each element y by x */
    sum = 0;
    for (i=0; i<d; i++) {
        z[i] = z[i]- y[i];
        sum += z[i]*z[i];
    }
    retval = exp(-0.5*gamma*sum);
    return retval;
}
double rbf_kernel_of_dist(double gamma, mwSize d, mwSize mA, mwSize mB, double *A, double *B){
    mwSize i, j;
    
    double sum; 
    sum = 0;
    for(i = 0; i<mA; i++)
        for(j = 0 ; j<mB;j++){
           sum += rbf_kernel_compute(gamma, d, &A[i], &B[j]);
        }
    return sum;
}
double normdiff_of_dist(double gamma, mwSize d, mwSize mA, mwSize mB, double *A, double *B){
    double sAA, sBB, sAB, result;
    sAA = rbf_kernel_of_dist(gamma, d, mA, mA, A, A); 
    sAB = rbf_kernel_of_dist(gamma, d, mA, mB, A, B); 
    sBB = rbf_kernel_of_dist(gamma, d, mB, mB, B, B); 
    result = sAA/(mA*mA) + sBB/(mB*mB) - 2*sAB/(mA*mB);
    return result;
}
/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double gamma;              /* input scalar */
    double *A;               /* 1xN input matrix */
    double *B;               /* 1xN input matrix */
    size_t nrows, nA, mA, nB, mB, d;                   /* size of matrix */
    double *outMatrix;              /* output matrix */

    /* check for proper number of arguments */
    if(nrhs!=3) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","Two inputs required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","One output required.");
    }
     
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
    /* make sure the third input argument is scalar */
    if( !mxIsDouble(prhs[2]) || 
         mxIsComplex(prhs[2]) ||
         mxGetNumberOfElements(prhs[2])!=1 ) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notScalar","Input multiplier must be a scalar.");
    }
    mA  = mxGetM(prhs[0]);
    nA  = mxGetN(prhs[0]);
    mB  = mxGetM(prhs[1]);
    nB  = mxGetN(prhs[1]);
    
    /*check that both of the first and second argument are in the same size*/
    if(nA != nB) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector","First and Second Input must be a of the same size");
    }
    d = nA;
    /* get the value of the scalar input  */
    gamma = mxGetScalar(prhs[2]);

    /* create a pointer to the real data in the input matrix  */
    A = mxGetPr(prhs[0]);
    B = mxGetPr(prhs[1]);
    mexPrintf("dimension is %d",d);
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);

    /* get a pointer to the real data in the output matrix */
    outMatrix = mxGetPr(plhs[0]);

    /* call the computational routine */
    outMatrix[0] = normdiff_of_dist(gamma, d, mA, mB, A, B);
}