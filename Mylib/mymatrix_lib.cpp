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
void inverseMatrix(ptrdiff_t dim, double *matrix, double *invMatrix, int copydata)
{
	// matrix and invMatrix are columnwise.
	ptrdiff_t *IPIV, LWORK, INFO=0, i;
	double *WORK;
	//mexPrintf("entered inverseMatrix");
	IPIV = (ptrdiff_t*)mxMalloc((dim+1)*sizeof(ptrdiff_t));
	LWORK = dim*dim;
	WORK = (double*)mxMalloc(LWORK*sizeof(double));
    if (copydata){// if copydata = 0, it means data is already is in invMatrix
        for (i=0;i<dim*dim;i++){
            invMatrix[i] = matrix[i];
        }
    }
	//mexPrintf("before dgetrf");
    dgetrf(&dim, &dim, invMatrix, &dim, IPIV, &INFO);
	//mexPrintf("before dgetri");
	dgetri(&dim, invMatrix, &dim, IPIV, WORK, &LWORK, &INFO);
    mxFree(IPIV);
	mxFree(WORK);
    return;
}
void matrix_multiply(ptrdiff_t m, ptrdiff_t v, ptrdiff_t p, ptrdiff_t n,  double *A, double *B, double *C) {
    //double *A, *B, *C; /* pointers to input & output matrices*/
    //size_t m,n,p;      /* matrix dimensions */
    /* form of op(A) & op(B) to use in matrix multiplication */
    char *chn = "N";
    /* scalar values to use in dgemm */
    double one = 1.0, zero = 0.0;

    if (p != v) {
        mexErrMsgIdAndTxt("MATLAB:matrixMultiply:matchdims",
                "Inner dimensions of matrix multiply do not match.");
    }

    /* Pass arguments to Fortran by reference */
    dgemm(chn, chn, &m, &n, &p, &one, A, &m, B, &p, &zero, C, &m);
}
void matrix_multiply(ptrdiff_t m, ptrdiff_t v, ptrdiff_t p, ptrdiff_t n,  double *A, int transA, double *B, int transB, double *C) {
    //double *A, *B, *C; /* pointers to input & output matrices*/
    //size_t m,n,p;      /* matrix dimensions */
    /* form of op(A) & op(B) to use in matrix multiplication */
    ptrdiff_t rowA, colA, rowB, colB, ldA, ldB, ldC;
    
    char *chN = "N";
    char *chT = "T";
    char *chA= chN;
    char *chB= chN;
    if(transA)
        chA = chT;
    if(transB)
        chB = chT;
    /* scalar values to use in dgemm */
    double one = 1.0, zero = 0.0;
    colA     = v;
    rowA     = m;
    ldA      = m;
    ldC      = m;
    if(transA){
        colA = m;
        rowA = v;
        ldC  = v;
    }
    ldB      = p;
    rowB     = p;
    colB     = n;
    if(transB){
        rowB = n;
        colB = p;
    }
    if (colA != rowB) {
        mexErrMsgIdAndTxt("MATLAB:matrixMultiply:matchdims",
                "Inner dimensions of matrix multiply do not match.");
    }
    /* Pass arguments to Fortran by reference */
    dgemm(chA, chB, &rowA, &colB, &colA, &one, A, &ldA, B, &ldB, &zero, C, &ldC);
}
void submatrix_multiply(ptrdiff_t dimsubA_1, ptrdiff_t dimsubA_2, ptrdiff_t dimB_1, ptrdiff_t dimB_2,  double *subA, ptrdiff_t leadingdim_A, double *B, double *C) {
    //double *A, *B, *C; /* pointers to input & output matrices*/
    //size_t m,n,p;      /* matrix dimensions */
    /* form of op(A) & op(B) to use in matrix multiplication */
    char *NoTrans = "N";
    /* scalar values to use in dgemm */
    double one = 1.0, zero = 0.0;

    if (dimsubA_2 != dimB_1) {
        mexErrMsgIdAndTxt("MATLAB:matrixMultiply:matchdims",
                "Inner dimensions of matrix multiply do not match.");
    }
    /* Pass arguments to Fortran by reference */
    dgemm(NoTrans, NoTrans, &dimsubA_1, &dimB_2, &dimB_1, &one, subA, &leadingdim_A, B, &dimB_1, &zero, C, &dimsubA_1);
}
void submat_mult_alphbeta(ptrdiff_t stidx_A_i, ptrdiff_t enidx_A_i, ptrdiff_t stidx_A_j, ptrdiff_t enidx_A_j, 
                          ptrdiff_t nrowA, ptrdiff_t ncolA, double alpha, double *A, 
                          ptrdiff_t nrowB, ptrdiff_t ncolB, double *B, double beta, double *C) {
    //double *A, *B, *C; /* pointers to input & output matrices*/
    //size_t m,n,p;      /* matrix dimensions */
    /* form of op(A) & op(B) to use in matrix multiplication */
    char *NoTrans = "N";
    /* scalar values to use in dgemm */
    //double one = 1.0, zero = 0.0;
    ptrdiff_t  leadingdim_A = nrowA;
    double myalpha = alpha;
    double mybeta  = beta;
    double *subA = A + stidx_A_j*nrowA + stidx_A_i;// check if it is correct?
    ptrdiff_t  dimsubA_1 = enidx_A_i-stidx_A_i;
    ptrdiff_t  dimsubA_2 = enidx_A_j-stidx_A_j;
    if (dimsubA_2 != nrowB) {
        mexPrintf("dimsubA_2:%d,!= nrowB :%d", dimsubA_2, nrowB);
        mexErrMsgIdAndTxt("MATLAB:matrixMultiply:matchdims",
                "Inner dimensions of matrix multiply do not match.");
    }
    //mexPrintf("dimsubA_1:%d , nColB: %d, nrowB:%d \n", dimsubA_1, ncolB, nrowB); 
    /* Pass arguments to Fortran by reference */
    dgemm(NoTrans, NoTrans, &dimsubA_1, &ncolB, &nrowB, &myalpha, subA, &leadingdim_A, B, &nrowB, &mybeta, C, &dimsubA_1);
}
double norm_with_respect(mwSize d, double * vec, double* mat){
    double result;
    double *medres = new double[d];
    matrix_multiply(d, d, d, 1, mat, vec, medres);
    matrix_multiply(1, d, d, 1, vec, medres, &result);
    delete medres;
    return result;
}
// In the following A must be a real symmetric positive semidefinite matrix, 
// it returns result in xb. Also, it gives xb as input. 
int solve_linear(ptrdiff_t nA, double *A, double *xb){
    
    ptrdiff_t info;
    const ptrdiff_t nrhs=1;
    ptrdiff_t *ipvs = new ptrdiff_t[nA];
    
    dgetrf(&nA, &nA, A, &nA, ipvs, &info);
    if(info) return info;
    

    dgetrs("T", &nA, &nrhs, A, &nA, ipvs, xb, &nA, &info);
    if(info) return info;

    return (int)info;
}
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
// void matrix_divide(size_t m, size_t p, size_t v, size_t n, double *A, double *B   ){
//     //double *A, *B;    /* pointers to input matrices */
//     double *A2, *B2;  /* in/out arguments to DGESV*/
//     //size_t m,n,p;     /* matrix dimensions */ 
//     mwSignedIndex *iPivot;   /* inputs to DGESV */
//     mxArray  *Awork, *mxPivot;
//     mwSignedIndex info, dims[2];
//  
//     /* Validate input arguments */
//     if (p != v) {
//         mexErrMsgIdAndTxt("MATLAB:matrixDivide:matchdims",
//             "Inner dimensions of matrices do not match.");
//     }
//     if (p != m) {
//         mexErrMsgIdAndTxt("MATLAB:matrixDivide:square",
//             "LAPACK function requires input matrix 1 must be square.");
//     }
//     if (n != 1) {
//         mexErrMsgIdAndTxt("MATLAB:matrixDivide:zerodivide",
//             "For this example input matrix 2 must be a column vector.");  
//     }
// 
//     /* DGESV works in-place, so we copy the inputs first. */
//     Awork   = mxCreateDoubleMatrix(m, p, mxREAL);
//     A2      = mxGetPr(Awork);
//     plhs[0] = mxCreateDoubleMatrix(p, n, mxREAL);
//     B2      = mxGetPr(plhs[0]);
//     memcpy(A2, A, m*p*mxGetElementSize(prhs[0]));
//     memcpy(B2, B, p*n*mxGetElementSize(prhs[1]));
//   
//     /* Create inputs for DGESV */
//     dims[0] = m;
//     dims[1] = p;
//     mxPivot = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
//     iPivot  = (mwSignedIndex*)mxGetData(mxPivot);
//   
//     /* Call LAPACK */
//     dgesv(&m,&n,A2,&m,iPivot,B2,&p,&info);
//     /* plhs[0] now holds X */
//   
//     mxDestroyArray(Awork);
//     mxDestroyArray(mxPivot);
//     
//     return; 
// }