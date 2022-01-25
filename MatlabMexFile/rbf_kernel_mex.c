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
double rbf_kernel_compute(double gamma, double *y, double *z, mwSize n)
{
    mwSize i;
    double sum, retval;
    /* multiply each element y by x */
    sum = 0;
    for (i=0; i<n; i++) {
        z[i] = z[i]- y[i];
        sum += z[i]*z[i];
    }
    retval = exp(-0.5*gamma*sum);
    return retval;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double gamma;              /* input scalar */
    double *inMatrix_1;               /* 1xN input matrix */
    double *inMatrix_2;               /* 1xN input matrix */
    size_t nrows;                   /* size of matrix */
    double *outMatrix;              /* output matrix */

    /* check for proper number of arguments */
    if(nrhs!=3) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","Two inputs required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","One output required.");
    }
    /* make sure the first input argument is scalar */
    /*if( !mxIsDouble(prhs[0]) || 
         mxIsComplex(prhs[0]) ||
         mxGetNumberOfElements(prhs[0])!=1 ) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notScalar","Input multiplier must be a scalar.");
    }*/
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
    /* check that number of rows in first input argument is 1 */
    if(mxGetN(prhs[0])!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector","Input must be a row vector.");
    }
    /* check that number of rows in second input argument is 1 */
    if(mxGetN(prhs[1])!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector","Input must be a row vector.");
    }
    /*check that both of the first and second argument are in the same size*/
    if(mxGetM(prhs[1]) != mxGetM(prhs[0])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector","First and Second Input must be a of the same size");
    }
    /* get the value of the scalar input  */
    gamma = mxGetScalar(prhs[2]);

    /* create a pointer to the real data in the input matrix  */
    inMatrix_1 = mxGetPr(prhs[0]);
    inMatrix_2 = mxGetPr(prhs[1]);

    /* get dimensions of the input matrix */
    nrows = mxGetM(prhs[1]);

    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);

    /* get a pointer to the real data in the output matrix */
    outMatrix = mxGetPr(plhs[0]);

    /* call the computational routine */
    outMatrix[0] = rbf_kernel_compute(gamma,inMatrix_1, inMatrix_2,(mwSize)nrows);
}