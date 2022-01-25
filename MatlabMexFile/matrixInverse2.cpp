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
#include "lapack.h"
#include "blas.h"
void inverseMatrix(ptrdiff_t dim, double *matrix, double *invMatrix)
{
	// matrix and invMatrix are columnwise.
	ptrdiff_t *IPIV, LWORK, INFO=0, i;
	double *WORK;
	mexPrintf("entered inverseMatrix");
	IPIV = (ptrdiff_t*)mxMalloc((dim+1)*sizeof(ptrdiff_t));
	LWORK = dim*dim;
	WORK = (double*)mxMalloc(LWORK*sizeof(double));
    for (i=0;i<dim*dim;i++){
		invMatrix[i] = matrix[i];
	}
	mexPrintf("before dgetrf");
    dgetrf(&dim, &dim, invMatrix, &dim, IPIV, &INFO);
	mexPrintf("before dgetri");
	dgetri(&dim, invMatrix, &dim, IPIV, WORK, &LWORK, &INFO);
    mxFree(IPIV);
	mxFree(WORK);
    return;
}
    
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
	inverseMatrix(row, xValues, mxGetPr(plhs[0]));
	
	return;
}
