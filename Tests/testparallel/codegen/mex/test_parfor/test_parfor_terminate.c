/*
 * test_parfor_terminate.c
 *
 * Code generation for function 'test_parfor_terminate'
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "test_parfor.h"
#include "test_parfor_terminate.h"
#include "test_parfor_data.h"

/* Function Definitions */
void test_parfor_atexit(void)
{
  emlrtStack st = { NULL, NULL, NULL };

  emlrtCreateRootTLS(&emlrtRootTLSGlobal, &emlrtContextGlobal, NULL, 1);
  st.tls = emlrtRootTLSGlobal;
  emlrtEnterRtStackR2012b(&st);
  emlrtLeaveRtStackR2012b(&st);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

void test_parfor_terminate(void)
{
  emlrtStack st = { NULL, NULL, NULL };

  st.tls = emlrtRootTLSGlobal;
  emlrtLeaveRtStackR2012b(&st);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

/* End of code generation (test_parfor_terminate.c) */
