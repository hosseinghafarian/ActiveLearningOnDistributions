/*=========================================================
 * matrixDivide.c - Example for illustrating how to use 
 * LAPACK within a C MEX-file.
 *
 * X = matrixDivide(A,B) computes the solution to a 
 * system of linear equations A * X = B
 * using LAPACK routine DGESV, where 
 * A is an N-by-N matrix  
 * X and B are N-by-1 matrices.
 *
 * This is a MEX-file for MATLAB.
 * Copyright 2009-2010 The MathWorks, Inc.
 *=======================================================*/

#if !defined(_WIN32)
#define dgesv dgesv_
#endif

#include "mex.h"
#include "mymatrix_lib.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *xData;
	double *xValues, *outArray, *invMatrix;
	int i,j;
	int rowLen, colLen;
	size_t row;
    
    xData = prhs[0];
    xValues = mxGetPr(xData);
	rowLen = mxGetM(xData);
	colLen = mxGetN(xData);
	row = rowLen;
    plhs[0] = mxCreateDoubleMatrix(rowLen, colLen, mxREAL); 
    outArray = mxGetPr(plhs[0]);
	inverseMatrix(row, xValues, mxGetPr(plhs[0]),1);
	
	return;
}
