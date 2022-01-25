/*
 * test_parfor.c
 *
 * Code generation for function 'test_parfor'
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "test_parfor.h"
#include "rand.h"
#include "fft.h"
#include "test_parfor_data.h"

/* Function Definitions */
void test_parfor(const emlrtStack *sp, real_T a[2560])
{
  int32_T i0;
  real_T r[2560];
  int32_T i;
  creal_T dc0;
  for (i0 = 0; i0 < 2560; i0++) {
    a[i0] = 1.0;
  }

  b_rand(r);
  i = 0;
  while (i < 10) {
    dc0 = fft(r[i]);
    for (i0 = 0; i0 < 256; i0++) {
      a[i + 10 * i0] = dc0.re;
    }

    i++;
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b(sp);
    }
  }
}

/* End of code generation (test_parfor.c) */
