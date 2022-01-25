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
void inverseMatrix(ptrdiff_t dim, double *matrix, double *invMatrix, int copydata);
void matrix_multiply(ptrdiff_t m, ptrdiff_t v, ptrdiff_t p, ptrdiff_t n,  double *A, double *B, double *C);
void matrix_multiply(ptrdiff_t m, ptrdiff_t v, ptrdiff_t p, ptrdiff_t n,  double *A, int transA, double *B, int transB, double *C);
/*
 * LAPACK within a C MEX-file.
 *
 * X = matrixDivide(A,B) computes the solution to a 
 * system of linear equations A * X = B
 * using LAPACK routine DGESV, where 
 * A is an N-by-N matrix  
 * X and B are N-by-1 matrices.
*/
// the following function can be changed in such a way that B :N-by-M, but I didn't do till now. 
void matrix_divide(size_t m, size_t p, size_t v, size_t n, double *A, double *B   );
void submat_mult_alphbeta(ptrdiff_t stidx_A_i, ptrdiff_t enidx_A_i, ptrdiff_t stidx_A_j, ptrdiff_t enidx_A_j, 
                          ptrdiff_t nrowA, ptrdiff_t ncolA, double alpha, double *A, 
                          ptrdiff_t nrowB, ptrdiff_t ncolB, double *B, double beta, double *C);
int solve_linear(ptrdiff_t nA, double *A, double *xb);
double norm_with_respect(mwSize d, double * vec, double* mat);