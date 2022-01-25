#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>
#include "smm.h"


Kernel::Kernel(int l, int n, smm_node * const * x_, int *gl_, int *cum_gl_, const smm_parameter& param)
:max_index(param.max_index),smm_mode(param.smm_mode),input_type(param.input_type),kernel_type(param.kernel_type), l2_kernel(param.l2_kernel), degree(param.degree), degree2(param.degree2), gamma(param.gamma), gamma2(param.gamma2), coef0(param.coef0), coef02(param.coef02)
{
	px = py = NULL;
	CC = DD = NULL;

	if(smm_mode == SVM) // svm mode
	{
		switch(kernel_type)
		{
			case LINEAR:
				kernel_function = &Kernel::kernel_linear;
				break;
			case POLY:
				kernel_function = &Kernel::kernel_poly;
				break;
			case RBF:
				kernel_function = &Kernel::kernel_rbf;
				break;
			case SIGMOID:
				kernel_function = &Kernel::kernel_sigmoid;
				break;
			case PRECOMPUTED:
				kernel_function = &Kernel::kernel_precomputed;
				break;
		}	
	}
	else if(smm_mode == SMM) // smm mode
	{
		switch(kernel_type)
		{
			case LINEAR:
				emb_kernel_function = &Kernel::kernel_linear;
				break;
			case POLY:
				emb_kernel_function = &Kernel::kernel_poly;
				break;
			case RBF:
				emb_kernel_function = &Kernel::kernel_rbf;
				break;
			case SIGMOID:
				emb_kernel_function = &Kernel::kernel_sigmoid;
				break;
		}		

		switch(l2_kernel)
		{
			case LINEAR:
				if(input_type == EMPIRICAL)
					kernel_function = &Kernel::l2_kernel_linear_empirical;
				else if(input_type == DISTRIBUTION)
				{
					if(kernel_type == LINEAR)
						kernel_function = &Kernel::l2_kernel_linear_linear;
					else if(kernel_type == POLY && degree == 2)
						kernel_function = &Kernel::l2_kernel_linear_poly2;
					else if(kernel_type == POLY && degree == 3)
						kernel_function = &Kernel::l2_kernel_linear_poly3;
					else if(kernel_type == RBF)
						kernel_function = &Kernel::l2_kernel_linear_rbf;
					
				}

				break;
			case POLY:
				kernel_function = &Kernel::l2_kernel_poly;
				break;
			case RBF:
				kernel_function = &Kernel::l2_kernel_rbf;
				break;
			case SIGMOID:
				kernel_function = &Kernel::l2_kernel_sigmoid;
				break;
		}		

		if(l2_kernel == POLY || l2_kernel == RBF || l2_kernel == SIGMOID)
		{
			if(input_type == EMPIRICAL)
				lin_kernel_function = &Kernel::l2_kernel_linear_empirical;
			else if(input_type == DISTRIBUTION)
			{
				if(kernel_type == LINEAR)
					lin_kernel_function = &Kernel::l2_kernel_linear_linear;
				else if(kernel_type == POLY && degree == 2)
					lin_kernel_function = &Kernel::l2_kernel_linear_poly2;
				else if(kernel_type == POLY && degree == 3)
					lin_kernel_function = &Kernel::l2_kernel_linear_poly3;
				else if(kernel_type == RBF)
					lin_kernel_function = &Kernel::l2_kernel_linear_rbf;
			}
		}

		if(input_type == DISTRIBUTION && kernel_type == RBF)
		{					
			px = Malloc(double,max_index);
			CC = Malloc(double *,max_index);
	
			for(int i=0;i<max_index;i++)
				CC[i] = Malloc(double,max_index);
		}
		else if(input_type == DISTRIBUTION && kernel_type == POLY)
		{
			if(degree == 2 || degree == 3)
			{
				px = Malloc(double,max_index);
				py = Malloc(double,max_index);
				CC = Malloc(double *,max_index);
				DD = Malloc(double *,max_index);

				for(int i=0;i<max_index;i++)
				{
					CC[i] = Malloc(double,max_index);
					DD[i] = Malloc(double,max_index);
				}
			}
			else	
			{
				fprintf(stderr,"The SMM with distribution input type for polynomial kernel of degree %d is not yet supported.\n",degree);
				exit(1);
			}
		}

		clone(gl,gl_,l);
		clone(cum_gl,cum_gl_,l);				

	}

	clone(x,x_,n);

	// precompute the norm for embedding RBF kernel
	if(kernel_type == RBF && (smm_mode == SVM || input_type == EMPIRICAL))
	{
		x_square = new double[n];
		for(int i=0;i<n;i++)
			x_square[i] = dot(x[i],x[i]);
	}
	else
		x_square = 0;

	// precompute the norm for level-2 RBF kernel
	if(l2_kernel == RBF)
	{
		xx_square = new double[l];
		for(int i=0;i<l;i++)
			xx_square[i] = (this->*lin_kernel_function)(i,i);
	}
	else
		xx_square = 0;
}

Kernel::~Kernel()
{
	delete[] x;
	delete[] gl;
	delete[] cum_gl;
	delete[] x_square;
	delete[] xx_square;
	delete[] px;
	delete[] CC;
	delete[] py;
	delete[] DD;
}

double Kernel::dot(const smm_node *px, const smm_node *py)
{
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
	return sum;
}

double Kernel::mahalanobis(double *px, double **C, int n)
{
	int i,j,k;
	double sum;
	double *diag, *sol;

	diag = Malloc(double,n);
	sol = Malloc(double,n);

	// Cholesky decomposition C = L*L';
	for(i=0;i<n;i++)
	{
		for(j=i;j<n;j++)
		{			
			for(sum=C[i][j],k=i-1;k>=0;k--) sum -= C[i][k]*C[j][k];
			if(i==j)
			{
				if(sum <= 0.0) // not positive definite
				{					
					fprintf(stderr,"Cholesky: the matrix is not positive definite.\n");
					exit(1);
				}

				diag[i] = sqrt(sum);
			}else
				C[j][i] = sum/diag[i];
		}			
	}

	// solve L*y = px, storing y in sol
	for(i=0;i<n;i++)
	{
		for(sum=px[i],k=i-1;k>=0;k--)
			sum -= C[i][k]*sol[k];
		sol[i]=sum/diag[i]; 		
	}

	// solve L'*x = y, storing x in sol
	for(i=n-1;i>=0;i--)
	{
		for(sum=sol[i],k=i+1;k<n;k++)
			sum -= C[k][i]*sol[k];
		sol[i]=sum/diag[i];
	}

	// compute the quadratic form
	sum=0;
	for(i=0;i<n;i++)
		sum += px[i]*sol[i];		

	delete[] diag;
	delete[] sol;

	return sum;
}

double Kernel::determinant(double **C, int n)
{
	int i,j,i_count,j_count,count;
	double **a, det=0;

	if(n==1) return C[0][0];
	if(n==2) return (C[0][0]*C[1][1] - C[0][1]*C[1][0]);

	a = Malloc(double *,n);
	for(i=0;i<n;i++) a[i] = Malloc(double,n);

	for(count=0;count<n;count++)
	{
		// array of minors
		i_count=0;
		for(i=1;i<n;i++)
		{
			j_count=0;
			for(j=0;j<n;j++)
			{
				if(j==count) continue;
				a[i_count][j_count]=C[i][j];
				j_count++;
			}
			i_count++;
		}		
		det += pow(-1,count)*C[0][count]*determinant(a,n-1);
	}

	for(i=0;i<n;i++) delete[] a[i];
	delete[] a;

	return det;
}

double Kernel::trace_mm(double **A, double **B, int n)
{
	double sum=0;
	for (int i=0;i<n;i++)
		for(int j=0;j<n;j++)
	       		sum += A[i][j]*B[j][i];

	return sum;
}

double Kernel::vmv_multiplication(double *a, double **B, double *c, int n)
{
	double sum=0,ssum;
	for(int i=0;i<n;i++)
	{
		ssum=0;
		for(int j=0;j<n;j++)		
			ssum += a[j]*B[j][i];
		sum += ssum*c[i];
	}		

	return sum;
}

double Kernel::vmmv_multiplication(double *a, double **B, double **D, double *c, int n)
{
	int i;
	double sum=0;
	double *u, *v;
	
	u = Malloc(double,n);
	v = Malloc(double,n);

	for(i=0;i<n;i++)
	{
		u[i]=0; v[i]=0;
		for(int j=0;j<n;j++)
		{
			u[i] += a[j]*B[j][i];
			v[i] += D[i][j]*c[j];
		}
	}

	for(i=0;i<n;i++)
		sum += u[i]*v[i];

	delete[] u; 
	delete[] v;

	return sum;
}

int Kernel::nearlyEqual(double a, double b, double epsilon)
{
	double absA = fabs(a);
	double absB = fabs(b);
	double diff = fabs(a - b);
	
	if (a == b) { // shortcut, handles infinities
		return true;
	} else if (a * b == 0) { // a or b or both are zero
		// relative error is not meaningful here
		return diff < (epsilon * epsilon);
	} else { // use relative error
		return diff / (absA + absB) < epsilon;
	}
}

double Kernel::k_function(const smm_node *x, const smm_node *y,
			  const smm_parameter& param)
{
	switch(param.kernel_type)
	{
		case LINEAR:
			return dot(x,y);
		case POLY:
			return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
		case RBF:
		{
			double sum = 0;
			while(x->index != -1 && y->index !=-1)
			{
				if(x->index == y->index)
				{
					double d = x->value - y->value;
					sum += d*d;
					++x;
					++y;
				}
				else
				{
					if(x->index > y->index)
					{	
						sum += y->value * y->value;
						++y;
					}
					else
					{
						sum += x->value * x->value;
						++x;
					}
				}
			}

			while(x->index != -1)
			{
				sum += x->value * x->value;
				++x;
			}

			while(y->index != -1)
			{
				sum += y->value * y->value;
				++y;
			}
			
			return exp(-param.gamma*sum);
		}
		case SIGMOID:
			return tanh(param.gamma*dot(x,y)+param.coef0);
		case PRECOMPUTED:  //x: test (validation), y: SM
			return x[(int)(y->value)].value;
		default:
			return 0;  // Unreachable 
	}
}

double Kernel::k_function(const smm_node **x, int n, const smm_node **y, int m,
			  const smm_parameter& param)
{
	int i,j,ii,jj;
	double sum,k_value=0,xs=0,ys=0;
	
	if(param.smm_mode == SMM)
	{
		if(param.input_type == EMPIRICAL)
		{
			// compute the sum of the embedding kernels
			switch(param.kernel_type)
			{
				case LINEAR:
					for(i=0;i<n;i++)
						for(j=0;j<m;j++)
							k_value += dot(x[i],y[j]);

					k_value = k_value/(n*m);

					if(param.l2_kernel == RBF) // for level-2 RBF kernel
					{
						xs=0; ys=0;
						for(i=0;i<n;i++)
							for(j=0;j<n;j++)
								xs += dot(x[i],x[j]);

						for(i=0;i<m;i++)
							for(j=0;j<m;j++)
								ys += dot(y[i],y[j]);

						xs = xs/(n*n); ys = ys/(m*m);
					}
					break;
				case POLY:
					for(i=0;i<n;i++)
						for(j=0;j<m;j++)
							k_value += powi(param.gamma*dot(x[i],y[j])+param.coef0,param.degree);
	
					k_value = k_value/(n*m);

					if(param.l2_kernel == RBF) // for level-2 RBF kernel
					{
						xs=0; ys=0;
						for(i=0;i<n;i++)
							for(j=0;j<n;j++)
								xs += powi(param.gamma*dot(x[i],x[j])+param.coef0,param.degree);

						for(i=0;i<m;i++)
							for(j=0;j<m;j++)
								ys += powi(param.gamma*dot(y[i],y[j])+param.coef0,param.degree);

						xs = xs/(n*n); ys = ys/(m*m);
					}
					break;
				case RBF:
					for(i=0;i<n;i++)
					{
						for(j=0;j<m;j++)
						{
							sum = 0;
							ii = jj = 0;
							while(x[i][ii].index != -1 && y[j][jj].index !=-1)
							{
								if(x[i][ii].index == y[j][jj].index)
								{
									double d = x[i][ii].value - y[j][jj].value;
									sum += d*d;
									++ii;
									++jj;
								}
								else
								{
									if(x[i][ii].index > y[j][jj].index)
									{	
										sum += y[j][jj].value * y[j][jj].value;
										++jj;
									}
									else
									{
										sum += x[i][ii].value * x[i][ii].value;
										++ii;
									}
								}
							}
	
							while(x[i][ii].index != -1)
							{
								sum += x[i][ii].value * x[i][ii].value;
								++ii;
							}
	
							while(y[j][jj].index != -1)
							{
								sum += y[j][jj].value * y[j][jj].value;
								++jj;
							}
				
							k_value += exp(-0.5*param.gamma*sum);							
						}
					}
	
					k_value = k_value/(n*m);

					if(param.l2_kernel == RBF) // for level-2 RBF kernel
					{
						xs=0; ys=0;
						
						// compute xs
						for(i=0;i<n;i++) 
						{
							for(j=0;j<n;j++)
							{
								sum = 0;
								ii = jj = 0;
								while(x[i][ii].index != -1 && x[j][jj].index !=-1)
								{
									if(x[i][ii].index == x[j][jj].index)
									{
										double d = x[i][ii].value - x[j][jj].value;
										sum += d*d;
										++ii;
										++jj;
									}
									else
									{
										if(x[i][ii].index > x[j][jj].index)
										{	
											sum += x[j][jj].value * x[j][jj].value;
											++jj;
										}
										else
										{
											sum += x[i][ii].value * x[i][ii].value;
											++ii;
										}
									}
								}
		
								while(x[i][ii].index != -1)
								{
									sum += x[i][ii].value * x[i][ii].value;
									++ii;
								}
		
								while(x[j][jj].index != -1)
								{
									sum += x[j][jj].value * x[j][jj].value;
									++jj;
								}
					
								xs += exp(-0.5*param.gamma*sum);							
							}
						}
	
						// compute ys
						for(i=0;i<m;i++) 
						{
							for(j=0;j<m;j++)
							{
								sum = 0;
								ii = jj = 0;
								while(y[i][ii].index != -1 && y[j][jj].index !=-1)
								{
									if(y[i][ii].index == y[j][jj].index)
									{
										double d = y[i][ii].value - y[j][jj].value;
										sum += d*d;
										++ii;
										++jj;
									}
									else
									{
										if(y[i][ii].index > y[j][jj].index)
										{	
											sum += y[j][jj].value * y[j][jj].value;
											++jj;
										}
										else
										{
											sum += y[i][ii].value * y[i][ii].value;
											++ii;
										}
									}
								}
		
								while(y[i][ii].index != -1)
								{
									sum += y[i][ii].value * y[i][ii].value;
									++ii;
								}
		
								while(y[j][jj].index != -1)
								{
									sum += y[j][jj].value * y[j][jj].value;
									++jj;
								}
					
								ys += exp(-0.5*param.gamma*sum);							
							}
						}

						xs = xs/(n*n); ys = ys/(m*m);
					}
					break;

				case SIGMOID:
					for(i=0;i<n;i++)
						for(j=0;j<m;j++)
							k_value += tanh(param.gamma*dot(x[i],y[j])+param.coef0);
	
					k_value = k_value/(n*m);
					
					if(param.l2_kernel == RBF) // for level-2 RBF kernel
					{
						xs=0; ys=0;
						for(i=0;i<n;i++)
							for(j=0;j<n;j++)
								xs += tanh(param.gamma*dot(x[i],x[j])+param.coef0);

						for(i=0;i<m;i++)
							for(j=0;j<m;j++)
								ys += tanh(param.gamma*dot(y[i],y[j])+param.coef0);

						xs = xs/(n*n); ys = ys/(m*m);
					}

					break;
				default:
					return 0;
			}
				
		}
		else if(param.input_type == DISTRIBUTION)
		{
			if(n != m)
			{
				fprintf(stderr,"The covariance matrix dimension mismatches.\n");
				exit(1);
			}
			else if(n != param.max_index+1)
			{
				fprintf(stderr,"The problem dimension mismatches.\n");
				exit(1);
			}

			switch(param.kernel_type)
			{
				case LINEAR:
					
					// check if the distributions are identical
					int identical;

					if(n != m) identical=0;
					else
					{
						identical=1;
						for(ii=0;ii<n+1 && identical==1;ii++)
							for (jj=0;jj<param.max_index && x[ii][jj].index!=-1 && y[ii][jj].index!=-1; jj++) 
							{
								if(x[ii][jj].index != y[ii][jj].index)
								{
									identical = 0;
									break;
								}
								
								if(!nearlyEqual(x[ii][jj].value, y[ii][jj].value, 0.00000001))
								{
									identical = 0;
									break;
								}
							}
					}

					if(identical == 0)
						k_value = dot(x[0],y[0]);
					else
					{
						int jj;

						k_value = dot(x[0],y[0]);

						for(ii=1;ii<n+1;ii++)
							for(jj=0;x[ii][jj].index != -1;jj++)
							{
								if(x[ii][jj].index == ii)
								{
									k_value += x[ii][jj].value;
									break;
								}
								else if(x[ii][jj].index > ii)
									break;						
							}
					}

					if(param.l2_kernel == RBF) // for level-2 RBF kernel
					{
						xs=dot(x[0],x[0]); 
						ys=dot(y[0],y[0]);;
						for(ii=1;ii<n+1;ii++)
							for(jj=0;x[ii][jj].index != -1;jj++)
							{
								if(x[ii][jj].index == ii)
								{
									xs += x[ii][jj].value;
									break;
								}
								else if(x[ii][jj].index > ii)
									break;						
							}
					
						for(ii=1;ii<n+1;ii++)
							for(jj=0;y[ii][jj].index != -1;jj++)
							{
								if(y[ii][jj].index == ii)
								{
									ys += y[ii][jj].value;
									break;
								}
								else if(y[ii][jj].index > ii)
									break;						
							}
					}

					break; 
				case POLY:

					double *ppx, *ppy;
					double **Cx, **Cy;

					switch(param.degree)
					{
						case 2:
							ppx = Malloc(double,param.max_index);
							ppy = Malloc(double,param.max_index);
							Cx = Malloc(double *,param.max_index);
							Cy = Malloc(double *,param.max_index);
							for(ii=0;ii<param.max_index;ii++)
							{
								Cx[ii] = Malloc(double,param.max_index);
								Cy[ii] = Malloc(double,param.max_index);
							}

							for(ii=0;ii<param.max_index;ii++)
							{
								ppx[ii]=0; ppy[ii]=0;
								for(int jj=0;jj<param.max_index;jj++)
								{
									Cx[ii][jj]=0;
									Cy[ii][jj]=0;
								}
							}

							for(jj=0;jj<param.max_index && x[0][jj].index!=-1;jj++) 
								ppx[x[0][jj].index-1] = x[0][jj].value;
	
							for(jj=0;jj<param.max_index && y[0][jj].index!=-1;jj++) 
								ppy[y[0][jj].index-1] = y[0][jj].value;

							for(ii=1;ii<=param.max_index;ii++)
							{			
								for(jj=0;jj<param.max_index;jj++)
								{
									if(x[ii][jj].index == -1) break;
									Cx[ii-1][x[ii][jj].index-1] = x[ii][jj].value;
								}
							}

							for(ii=1;ii<=param.max_index;ii++)
							{			
								for(jj=0;jj<param.max_index;jj++)
								{
									if(y[ii][jj].index == -1) break;									
									Cy[ii-1][y[ii][jj].index-1] = y[ii][jj].value;
								}
							}

							k_value = powi(param.gamma*dot(x[0],y[0])+param.coef0,2) + trace_mm(Cx,Cy,param.max_index) + vmv_multiplication(ppx,Cy,ppx,param.max_index) + vmv_multiplication(ppy,Cx,ppy,param.max_index);

							if(param.l2_kernel == RBF) // for level-2 RBF kernel
							{
								xs=powi(param.gamma*dot(x[0],x[0])+param.coef0,2) + trace_mm(Cx,Cx,param.max_index) + 2*vmv_multiplication(ppx,Cx,ppx,param.max_index);
								ys=powi(param.gamma*dot(y[0],y[0])+param.coef0,2) + trace_mm(Cy,Cy,param.max_index) + 2*vmv_multiplication(ppy,Cy,ppy,param.max_index);
							}						

							delete[] ppx;
							delete[] ppy;

							for(ii=0;ii<param.max_index;ii++)
							{
								delete[] Cx[ii];
								delete[] Cy[ii];
							}

							delete[] Cx;
							delete[] Cy; 

							break;

						case 3:

							ppx = Malloc(double,param.max_index);
							ppy = Malloc(double,param.max_index);
							Cx = Malloc(double *,param.max_index);
							Cy = Malloc(double *,param.max_index);
							for(ii=0;ii<param.max_index;ii++)
							{
								Cx[ii] = Malloc(double,param.max_index);
								Cy[ii] = Malloc(double,param.max_index);
							}

							for(ii=0;ii<param.max_index;ii++)
							{
								ppx[ii]=0; ppy[ii]=0;
								for(int jj=0;jj<param.max_index;jj++)
								{
									Cx[ii][jj]=0;
									Cy[ii][jj]=0;
								}
							}

							for(jj=0;jj<param.max_index && x[0][jj].index!=-1;jj++) 
								ppx[x[0][jj].index-1] = x[0][jj].value;
	
							for(jj=0;jj<param.max_index && y[0][jj].index!=-1;jj++) 
								ppy[y[0][jj].index-1] = y[0][jj].value;

							for(ii=1;ii<=param.max_index;ii++)
							{			
								for(jj=0;jj<param.max_index;jj++)
								{
									if(x[ii][jj].index == -1) break;
									Cx[ii-1][x[ii][jj].index-1] = x[ii][jj].value;
								}
							}

							for(ii=1;ii<=param.max_index;ii++)
							{			
								for(jj=0;jj<param.max_index;jj++)
								{
									if(y[ii][jj].index == -1) break;
									Cy[ii-1][y[ii][jj].index-1] = y[ii][jj].value;
								}
							}


							k_value = powi(param.gamma*dot(x[0],y[0])+param.coef0,3) + 6*vmmv_multiplication(ppx,Cx,Cy,ppy,param.max_index) + 3*(dot(x[0],y[0])+param.coef0)*(trace_mm(Cx,Cy,param.max_index) + vmv_multiplication(ppx,Cy,ppx,param.max_index) + vmv_multiplication(ppy,Cx,ppy,param.max_index));

							if(param.l2_kernel == RBF) // for level-2 RBF kernel
							{
								xs=powi(param.gamma*dot(x[0],x[0])+param.coef0,3) + 6*vmmv_multiplication(ppx,Cx,Cx,ppx,param.max_index) + 3*(dot(x[0],x[0])+param.coef0)*(trace_mm(Cx,Cx,param.max_index) + 2*vmv_multiplication(ppx,Cx,ppx,param.max_index));
								ys=powi(param.gamma*dot(y[0],y[0])+param.coef0,3) + 6*vmmv_multiplication(ppy,Cy,Cy,ppy,param.max_index) + 3*(dot(y[0],y[0])+param.coef0)*(trace_mm(Cy,Cy,param.max_index) + 2*vmv_multiplication(ppy,Cy,ppy,param.max_index));
							}	

							delete[] ppx;
							delete[] ppy;
							
							for(ii=0;ii<param.max_index;ii++)
							{
								delete[] Cx[ii];
								delete[] Cy[ii];
							}

							delete[] Cx;
							delete[] Cy; 

							break;
						default: 
							return 0;
					}

					break;

				case RBF:
					
					int jj;
					double d_square, dt;
					double *p;
					double **D;					

					p = Malloc(double,param.max_index);
					D = Malloc(double *,param.max_index);
					
					for(ii=0;ii<param.max_index;ii++)
					{
						p[ii]=0;
						D[ii]=Malloc(double,param.max_index);
						for(jj=0;jj<param.max_index;jj++)
							D[ii][jj]=0;
					}

					for(jj=0;jj<param.max_index && x[0][jj].index!=-1;jj++) 
						p[x[0][jj].index-1] += x[0][jj].value;

					for(jj=0;jj<param.max_index && y[0][jj].index!=-1;jj++) 
						p[y[0][jj].index-1] -= y[0][jj].value;
					
					for(ii=1;ii<=param.max_index;ii++)
					{
						for(jj=0;jj<param.max_index && x[ii][jj].index != -1;jj++)
							D[ii-1][x[ii][jj].index-1] += x[ii][jj].value;

						for(jj=0;jj<param.max_index && y[ii][jj].index != -1;jj++)
							D[ii-1][y[ii][jj].index-1] += y[ii][jj].value;

						D[ii-1][ii-1] += 1.0/param.gamma;
					}							

					dt = powi(param.gamma,param.max_index)*determinant(D,param.max_index);
					d_square = mahalanobis(p, D, param.max_index);		
						
					k_value = exp(-0.5*d_square)/sqrt(dt);

					if(param.l2_kernel == RBF) // for level-2 RBF kernel
					{
						for(ii=1;ii<=param.max_index;ii++)
						{			
							for(jj=0;jj<param.max_index;jj++)
							{
								if(x[ii][jj].index == -1) break;
								D[ii-1][x[ii][jj].index-1] = 2*param.gamma*x[ii][jj].value + 1;
							}

						}

						xs = 1.0/sqrt(determinant(D,param.max_index));

						for(ii=1;ii<=param.max_index;ii++)
						{			
							for(jj=0;jj<param.max_index;jj++)
							{
								if(y[ii][jj].index == -1) break;
								D[ii-1][y[ii][jj].index-1] = 2*param.gamma*y[ii][jj].value + 1;
							}
	
						}

						ys = 1.0/sqrt(determinant(D,param.max_index));
					}

					delete[] p;
					for(ii=0;ii<param.max_index;ii++) delete[] D[ii];
					delete[] D;

					break;
				case SIGMOID: // not available for this input type
				default:
					return 0;
			}
		}

		// compute the level-2 kernel
		switch(param.l2_kernel)
		{
			case LINEAR:
				return k_value;
			case POLY:
				return powi(param.gamma2*k_value+param.coef02,param.degree2);
			case RBF:
				return exp(-param.gamma2*(xs+ys-2*k_value));
			case SIGMOID:
				return tanh(param.gamma2*k_value+param.coef02);
			default:
				return 0;
		}
	}
	else
	{
		fprintf(stderr,"The kernel evaluation is available only in SMM mode.\n");
		return 0;
	}
	
}