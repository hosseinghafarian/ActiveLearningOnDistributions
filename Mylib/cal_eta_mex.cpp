/*==========================================================
 * cal_eta_mex.cpp - 
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
void   print_vec(double *p, mwSize d){
    mwSize i;
	mexPrintf("[ ");
    for(i=0;i<d;i++)
        mexPrintf("%f,",p[i]);
	mexPrintf("]");
}
void   print_vec(mwSize *p, mwSize d){
    mwSize i;
	mexPrintf("[ ");
    for(i=0;i<d;i++)
        mexPrintf("%5d,",p[i]);
	mexPrintf("]");
}
void print_matrix(double *p, mwSize d, mwSize n){
    mwSize i,j;
    mexPrintf("\n[");
    for(i=0;i<d;i++){
        mexPrintf(";\n");
        for(j=0;j<n;j++)
           mexPrintf(",%7.4f",p[i+j*n]);
    }
    mexPrintf("]\n");
}
void validate_inputs(mwSize *n, mwSize* numL, mwSize*numU, int nlhs, mxArray *plhs[],
                                                    int nrhs, const mxArray *prhs[] ) { 
    mwSize mA;
    /* check for proper number of arguments */
    if(nrhs!=5) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","Seven inputs required.");
    }
    /* make sure the first input argument is type double */
    if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Input matrix must be type double.");
    }
    *n   = mxGetM(prhs[1]);
    mA   = mxGetN(prhs[1]);
    if ( mA!=*n){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Second argument must be an square matrix.");
    }
    mexPrintf("% 5d\n", *n);
    *numU= mxGetN(prhs[2]);
    *numL= mxGetM(prhs[0]);
    
    if (*numU != *n-*numL){
       mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Number of instances doesnot match with labeled+unlabeled instances."); 
    }
    
    /* make sure the second input argument is type double */
    if( !mxIsDouble(prhs[0])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","First argument must be a integer row with number of labeled instances.");
    }
    if( !mxIsDouble(prhs[2])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Third argument must be a row vector of integers");
    }
}
void cal_eta_c(double* out_etas, double *Yl, double* L, double* LL, mwSize n, mwSize numL, mwSize numU ){
    
    mwSize i,j;
    double *b = new double[n];
    mwSize *indexL = new mwSize[n];
    mwSize *indexU = new mwSize[n];
    mwSize *tmpindx= new mwSize[n];
    double *t1     = new double[n];
    double *t1D    = new double[n];
    double *DLus   = new double[n];
    double t1b;
    double bLus;
    double LslYl, t1DLus, t1Dt1, LusDLus;
    
    //mexPrintf("in cal_eta_c, n:%5d, numL:%5d, numU:%5d", n, numL, numU);
    
    for(j=0; j<numL;j++){
        indexL[j] = j;
        //mexPrintf("indexL[%d]:%d\n",j,indexL[j]);
    }
    //indexL[numL] = -1;
    //print_matrix(L, n, n);
    //mexPrintf("\n");
    //print_vec(indexL, numL);
    for(i = 0; i< numU;i++){
        mwSize indexS, iU_stpre, iU_enpre, iU_staft, iU_enaft;
        indexS = i + numL;
        //mexPrintf("\n indexS:%5d\n", indexS);
        mwSize p;
        // indexU
        p = 0;
        for(j=numL; j<n; j++){
           if(j!=indexS){
              indexU[p]  = j;
              tmpindx[p] = indexU[p]-numL;
              p++;
           }
        }   
        //print_vec(tmpindx, numU-1);
//         mexPrintf("\n");
        double a = LL[i*numU + i];
        //mexPrintf("a:%f\n", a);
        // copy first part of b
        mwSize sizeb = 0;
        for(j=0; j<numU-1;j++){
            b[j] = -LL[tmpindx[j]*numU+i];
            sizeb++;
        }
        // t1=Lul*Yl; 
        mwSize k;
        //mexPrintf("numU:%d, p:%d\n",numU,p);
        for(j=0;j<numU-1;j++){
            t1[j]=0;
            //mexPrintf("indexL[%d]:%d\n", 0,indexL[0]);
            for(k=0;k<numL;k++){
               //mexPrintf("indexL[%d]:%d, indexU[%d]:%d,", k,indexL[k], j,indexU[j]);
               //mexPrintf("L[indexL[k]*n+indexU[j]]:%f",L[indexL[k]*n+indexU[j]]);
               t1[j] += L[indexL[k]*n+indexU[j]]*Yl[indexL[k]];       
            }
//             mexPrintf("%f,",t1[j]);
//             mexPrintf("\n");
        }
        //print_vec(t1, sizeb);
        //mexPrintf("\n");
        //print_vec(b, sizeb);
//         //t1b = t1'*b';
//         //bLus = b*Lus;
        t1b = 0;bLus = 0;
        for(j=0;j<sizeb;j++){
            t1b += t1[j]*b[j];
            bLus+= L[indexS*n+indexU[j]]*b[j];
        }
        //mexPrintf("t1b:%f, bLus:%f\n", t1b, bLus);
        // t1D
        for(j=0;j<numU-1;j++){
            t1D[j] = 0;
            for(k=0; k<numU-1;k++)
               t1D[j] += t1[k]*LL[numU*tmpindx[j]+tmpindx[k]];
        }
        //print_vec(t1D, sizeb);
        // Dlus
        for(j=0;j<numU-1;j++){
            DLus[j] = 0;
            for(mwSize k=0; k<numU-1;k++)
              DLus[j] += L[indexS*n+indexU[k]]*LL[numU*tmpindx[j]+tmpindx[k]];
        }
        //print_vec(DLus, numU-1);
        // LslYl
        LslYl = 0;
        for(j=0;j<numL;j++){
            LslYl += L[indexL[j]*n+indexS]*Yl[j];
        }
        //mexPrintf("LslYl:%f",LslYl);
        // t1Dt1
        t1Dt1 = 0;
        for(j=0;j<numU-1;j++)
            t1Dt1 += t1D[j]*t1[j];
        //mexPrintf("t1Dt1:%f",t1Dt1);
        // LusDLus
        LusDLus = 0;
        for(j=0;j<numU-1;j++)
            LusDLus += DLus[j]*L[indexS*n+indexU[j]];
        //mexPrintf("LusDLus:%f",LusDLus);
        // t1DLus
        t1DLus = 0;
        for(j=0;j<sizeb;j++)
            t1DLus += DLus[j]*t1[j];
        //mexPrintf("t1DLus:%f",t1DLus);
        double t4, t5, t6;
        t4 = t1DLus - t1b*bLus/a;
        //mexPrintf("t4:%f",t4);
        t5 = 2*LslYl - 2*t4;
        //mexPrintf("t5:%f",t5);
        t6 = L[indexS*n+indexS]- t1Dt1 - LusDLus + t1b*t1b/a+bLus*bLus/a;
        //mexPrintf("t6:%f",t6);
        if (t5 >=0)
            out_etas[i] = t6+t5;
        else
            out_etas[i] = t6-t5;
        //mexPrintf("out_etas[i]:%f",out_etas[i]);
    }
}
/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double *Yl, *L, *LL;               /* 1xN input matrix */

    mwSize numL,n,numU;                   /* size of matrix */
    double *out_etas;                 /* output vector */
   
    //validate_inputs(&n, &numL, &numU, nlhs,plhs,nrhs,prhs );
     
    /* create a pointer to the real data in the input matrix  */
    Yl        = mxGetPr(prhs[0]);
    //print_vec(Yl, numL);
    L         = mxGetPr(prhs[1]);
    //print_matrix(L, n, n);
    LL        = mxGetPr(prhs[2]);
    numL      = mxGetScalar(prhs[4]);
    n         = mxGetScalar(prhs[3]);
    numU      = n-numL;
    //mexPrintf("\n, n:%5d, numU:%5d, numL:%5d \n", n,numU, numL);
    /* create the output matrix */
    plhs[0]   = mxCreateDoubleMatrix(1,      numU, mxREAL);

    /* get a pointer to the real data in the output matrix */
    out_etas  = mxGetPr(plhs[0]);
    //distance_matrix(d, nA, A, F_A, midx_A, idx_dF_A, nB, B, F_B, midx_B, idx_dF_B, gamma, outMatrix );
    cal_eta_c(out_etas, Yl, L, LL, n, numL, numU);
}