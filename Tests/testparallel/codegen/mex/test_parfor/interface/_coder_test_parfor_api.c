/*
 * _coder_test_parfor_api.c
 *
 * Code generation for function '_coder_test_parfor_api'
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "test_parfor.h"
#include "_coder_test_parfor_api.h"
#include "test_parfor_data.h"

/* Function Declarations */
static const mxArray *emlrt_marshallOut(const real_T u[2560]);

/* Function Definitions */
static const mxArray *emlrt_marshallOut(const real_T u[2560])
{
  const mxArray *y;
  static const int32_T iv0[2] = { 0, 0 };

  const mxArray *m0;
  static const int32_T iv1[2] = { 10, 256 };

  y = NULL;
  m0 = emlrtCreateNumericArray(2, iv0, mxDOUBLE_CLASS, mxREAL);
  mxSetData((mxArray *)m0, (void *)u);
  emlrtSetDimensions((mxArray *)m0, iv1, 2);
  emlrtAssign(&y, m0);
  return y;
}

void test_parfor_api(const mxArray *plhs[1])
{
  real_T (*a)[2560];
  emlrtStack st = { NULL, NULL, NULL };

  st.tls = emlrtRootTLSGlobal;
  a = (real_T (*)[2560])mxMalloc(sizeof(real_T [2560]));

  /* Invoke the target function */
  test_parfor(&st, *a);

  /* Marshall function outputs */
  plhs[0] = emlrt_marshallOut(*a);
}

/* End of code generation (_coder_test_parfor_api.c) */
