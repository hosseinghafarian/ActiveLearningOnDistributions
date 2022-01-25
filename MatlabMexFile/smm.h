#ifndef _LIBSMM_H
#define _LIBSMM_H

#define LIBSMM_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

extern int libsmm_version;

struct smm_node
{
	int index;
	double value;
};

struct smm_problem
{
	int l;
	int n;
	int *gl;
	int *cum_gl;
	double *y;
	double *wi;
	struct smm_node **x;
};

enum { SMM, SVM }; 	/* learning mode: SVM or SMM */
enum { EMPIRICAL, DISTRIBUTION };	/* input type */
enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

struct smm_parameter
{
	int smm_mode;
	int input_type;
	int svm_type;
	int kernel_type;
	int l2_kernel;
	int degree;	/* for poly */
	int degree2;	
	double gamma;	/* for poly/rbf/sigmoid */
	double gamma2;	
	double coef0;	/* for poly/sigmoid */
	double coef02;

	int max_index;  /* the dimensionality */

	/* these are for training only */
	double cache_size; /* in MB */
	double eps;	/* stopping criteria */
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
	int nr_weight;		/* for C_SVC */
	int *weight_label;	/* for C_SVC */
	double* weight;		/* for C_SVC */
	double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	double p;	/* for EPSILON_SVR */
	int shrinking;	/* use the shrinking heuristics */
	int probability; /* do probability estimates */
};

//
// smm_model
// 
struct smm_model
{
	struct smm_parameter param;	/* parameter */
	int nr_class;		/* number of classes, = 2 in regression/one class smm */
	int l;			/* total #SM */
	int n;			/* total samples */
	int *gl; 		/* the group size */
	int *cum_gl; 		/* the cumulative group size */
	struct smm_node **SM;		/* SMs (SM[l]) */
	double **sm_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */
	double *probA;		/* pariwise probability information */
	double *probB;

	/* for classification only */

	int *label;		/* label of each class (label[k]) */
	int *nSM;		/* number of SMs for each class (nSM[k]) */
				/* nSM[0] + nSM[1] + ... + nSV[k-1] = l */
	/* XXX */
	int free_sm;		/* 1 if smm_model is created by smm_load_model*/
				/* 0 if smm_model is created by smm_train */
};
int libsmm_version = LIBSMM_VERSION;
typedef float Qfloat;
typedef signed char schar;
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void (*smm_print_string) (const char *) = &print_string_stdout;
#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*smm_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache
{
public:
	Cache(int l,long int size);
	~Cache();

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	int get_data(const int index, Qfloat **data, int len);
	void swap_index(int i, int j);	
private:
	int l;
	long int size;
	struct head_t
	{
		head_t *prev, *next;	// a circular list
		Qfloat *data;
		int len;		// data[0,len) is cached in this entry
	};

	head_t *head;
	head_t lru_head;
	void lru_delete(head_t *h);
	void lru_insert(head_t *h);
};

Cache::Cache(int l_,long int size_):l(l_),size(size_)
{
	head = (head_t *)calloc(l,sizeof(head_t));	// initialized to 0
	size /= sizeof(Qfloat);
	size -= l * sizeof(head_t) / sizeof(Qfloat);
	size = max(size, 2 * (long int) l);	// cache must be large enough for two columns
	lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache()
{
	for(head_t *h = lru_head.next; h != &lru_head; h=h->next)
		free(h->data);
	free(head);
}

void Cache::lru_delete(head_t *h)
{
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h)
{
	// insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

int Cache::get_data(const int index, Qfloat **data, int len)
{
	head_t *h = &head[index];
	if(h->len) lru_delete(h);
	int more = len - h->len;

	if(more > 0)
	{
		// free old space
		while(size < more)
		{
			head_t *old = lru_head.next;
			lru_delete(old);
			free(old->data);
			size += old->len;
			old->data = 0;
			old->len = 0;
		}

		// allocate new space
		h->data = (Qfloat *)realloc(h->data,sizeof(Qfloat)*len);
		size -= more;
		swap(h->len,len);
	}

	lru_insert(h);
	*data = h->data;
	return len;
}

void Cache::swap_index(int i, int j)
{
	if(i==j) return;

	if(head[i].len) lru_delete(&head[i]);
	if(head[j].len) lru_delete(&head[j]);
	swap(head[i].data,head[j].data);
	swap(head[i].len,head[j].len);
	if(head[i].len) lru_insert(&head[i]);
	if(head[j].len) lru_insert(&head[j]);

	if(i>j) swap(i,j);
	for(head_t *h = lru_head.next; h!=&lru_head; h=h->next)
	{
		if(h->len > i)
		{
			if(h->len > j)
				swap(h->data[i],h->data[j]);
			else
			{
				// give up
				lru_delete(h);
				free(h->data);
				size += h->len;
				h->data = 0;
				h->len = 0;
			}
		}
	}
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
public:
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const = 0;
	virtual ~QMatrix() {}
};

class Kernel: public QMatrix {
public:
	Kernel(int l, int n, smm_node * const * x, int *gl, int *cum_gl, const smm_parameter& param);
	virtual ~Kernel();

	static double k_function(const smm_node *x, const smm_node *y,
				 const smm_parameter& param);

	static double k_function(const smm_node **x, int n, const smm_node **y, int m,
				 const smm_parameter& param);	
	
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const	// no so const...
	{		
		if(smm_mode == SVM)
			swap(x[i],x[j]);
		else if(smm_mode == SMM)
		{
			swap(gl[i],gl[j]);
			swap(cum_gl[i],cum_gl[j]);
		}

		if(x_square) swap(x_square[i],x_square[j]);
		if(xx_square) swap(xx_square[i],xx_square[j]);
	}
protected:

	double (Kernel::*kernel_function)(int i, int j) const;
	double (Kernel::*emb_kernel_function)(int i, int j) const;
	double (Kernel::*lin_kernel_function)(int i, int j) const;

private:
	const smm_node **x;
	int max_index;
	int *gl;
	int *cum_gl;
	double *x_square;
	double *xx_square;	
	double *px,*py;	
	double **CC,**DD;

	// smm_parameter
	const int smm_mode;
	const int input_type;
	const int kernel_type;
	const int l2_kernel;
	const int degree;
	const int degree2;
	const double gamma;
	const double gamma2;
	const double coef0;
	const double coef02;

	static double dot(const smm_node *px, const smm_node *py);
	static double mahalanobis(double *px, double **C, int n);
	static double determinant(double **C, int n);
	static double trace_mm(double **A, double **B, int n);
	static double vmv_multiplication(double *a, double **B, double *c, int n);
	static double vmmv_multiplication(double *a, double **B, double **D, double *c, int n);
	static int nearlyEqual(double a, double b, double epsilon);
	
	double kernel_linear(int i, int j) const
	{
		return dot(x[i],x[j]);
	}
	double kernel_poly(int i, int j) const
	{
		return powi(gamma*dot(x[i],x[j])+coef0,degree);
	}
	double kernel_rbf(int i, int j) const
	{
		return exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
	}
	double kernel_sigmoid(int i, int j) const
	{
		return tanh(gamma*dot(x[i],x[j])+coef0);
	}
	double kernel_precomputed(int i, int j) const
	{
		return x[i][(int)(x[j][0].value)].value;
	}
		
	double l2_kernel_linear_empirical(int i, int j) const
	{
		int ii,jj;
		double sum=0;

		for(ii=cum_gl[i]; ii<cum_gl[i]+gl[i]; ii++)
			for(jj=cum_gl[j]; jj<cum_gl[j]+gl[j]; jj++)
				sum += (this->*emb_kernel_function)(ii,jj);

		return sum/(gl[i]*gl[j]);
	}

	double l2_kernel_linear_linear(int i, int j) const
	{

		if(i != j)
			return dot(x[cum_gl[i]],x[cum_gl[j]]);
		else
		{
			double sum = dot(x[cum_gl[i]],x[cum_gl[j]]);		
	
			for(int ii=cum_gl[i]+1;ii<gl[i]-1;ii++)
				for(int jj=0;x[ii][jj].index != -1;jj++)
				{					
					if(cum_gl[i]+1+x[ii][jj].index == ii)
					{
						sum += x[ii][jj].value;
						break;
					}
					else if(cum_gl[i]+1+x[ii][jj].index > ii)
						break;						
				}

			return sum;
		}
	}	

	double l2_kernel_linear_rbf(int i, int j) const
	{
		int ii,jj;
		double d_square, dt;
						
		for(ii=0;ii<max_index;ii++)
		{
			px[ii]=0;
			for(jj=0;jj<max_index;jj++)
				CC[ii][jj]=0;
		}
		
		for(jj=0;jj<max_index && x[cum_gl[i]][jj].index!=-1;jj++)
			px[x[cum_gl[i]][jj].index-1] += x[cum_gl[i]][jj].value;
			
		for(jj=0;jj<max_index && x[cum_gl[j]][jj].index!=-1;jj++) 
			px[x[cum_gl[j]][jj].index-1] -= x[cum_gl[j]][jj].value;
		
		for(ii=1;ii<=max_index;ii++)
		{			
			for(jj=0;jj<max_index && x[cum_gl[i]+ii][jj].index != -1;jj++)
				CC[ii-1][x[cum_gl[i]+ii][jj].index-1] += x[cum_gl[i]+ii][jj].value;
			
			for(jj=0;jj<max_index && x[cum_gl[j]+ii][jj].index != -1;jj++)
				CC[ii-1][x[cum_gl[j]+ii][jj].index-1] += x[cum_gl[j]+ii][jj].value;

			CC[ii-1][ii-1] += 1.0/gamma;
		}

		dt = powi(gamma,max_index)*determinant(CC,max_index);
		d_square = mahalanobis(px, CC, max_index);				

		return exp(-0.5*d_square)/sqrt(dt);
	}	

	double l2_kernel_linear_poly2(int i, int j) const
	{	
		
		int ii,jj;
		for(ii=0;ii<max_index;ii++)
		{
			px[ii]=0; py[ii]=0;
			for(int jj=0;jj<max_index;jj++)
			{
				CC[ii][jj]=0;
				DD[ii][jj]=0;
			}
		}

		for(jj=0;jj<max_index && x[cum_gl[i]][jj].index!=-1;jj++) 
			px[x[cum_gl[i]][jj].index-1] = x[cum_gl[i]][jj].value;
	
		for(jj=0;jj<max_index && x[cum_gl[j]][jj].index!=-1;jj++) 
			py[x[cum_gl[j]][jj].index-1] = x[cum_gl[j]][jj].value;

		for(ii=1;ii<=max_index;ii++)
		{			
			for(jj=0;jj<max_index;jj++)
			{
				if(x[cum_gl[i]+ii][jj].index == -1) break;
				CC[ii-1][x[cum_gl[i]+ii][jj].index-1] = x[cum_gl[i]+ii][jj].value;
			}
		}

		for(ii=1;ii<=max_index;ii++)
		{			
			for(jj=0;jj<max_index;jj++)
			{
				if(x[cum_gl[j]+ii][jj].index == -1) break;				
				DD[ii-1][x[cum_gl[j]+ii][jj].index-1] = x[cum_gl[j]+ii][jj].value;
			}
		}
		
		return powi(gamma*dot(x[cum_gl[i]],x[cum_gl[j]])+coef0,2) + trace_mm(CC,DD,max_index) + vmv_multiplication(px,DD,px,max_index) + vmv_multiplication(py,CC,py,max_index);
	}

	double l2_kernel_linear_poly3(int i, int j) const
	{
		int ii,jj;
		for(ii=0;ii<max_index;ii++)
		{
			px[ii]=0; py[ii]=0;
			for(int jj=0;jj<max_index;jj++)
			{
				CC[ii][jj]=0;
				DD[ii][jj]=0;
			}
		}

		for(jj=0;jj<max_index && x[cum_gl[i]][jj].index!=-1;jj++) 
			px[x[cum_gl[i]][jj].index-1] = x[cum_gl[i]][jj].value;
	
		for(jj=0;jj<max_index && x[cum_gl[j]][jj].index!=-1;jj++) 
			py[x[cum_gl[j]][jj].index-1] = x[cum_gl[j]][jj].value;

		for(ii=1;ii<=max_index;ii++)
		{			
			for(jj=0;jj<max_index;jj++)
			{
				if(x[cum_gl[i]+ii][jj].index == -1) break;
				CC[ii-1][x[cum_gl[i]+ii][jj].index-1] = x[cum_gl[i]+ii][jj].value;
			}
		}

		for(ii=1;ii<=max_index;ii++)
		{			
			for(jj=0;jj<max_index;jj++)
			{
				if(x[cum_gl[j]+ii][jj].index == -1) break;
				DD[ii-1][x[cum_gl[j]+ii][jj].index-1] = x[cum_gl[j]+ii][jj].value;
			}
		}

		return powi(gamma*dot(x[cum_gl[i]],x[cum_gl[j]])+coef0,3) + 6*vmmv_multiplication(px,CC,DD,py,max_index) + 3*(dot(x[cum_gl[i]],x[cum_gl[j]])+coef0)*(trace_mm(CC,DD,max_index) + vmv_multiplication(px,DD,px,max_index) + vmv_multiplication(py,CC,py,max_index));
	}

	double l2_kernel_rbf(int i, int j) const
	{
		return exp(-gamma2*(xx_square[i]+xx_square[j]-2*(this->*lin_kernel_function)(i,j)));
	}
	
	double l2_kernel_poly(int i, int j) const
	{		
		return powi(gamma2*(this->*lin_kernel_function)(i,j)+coef02,degree2);
	}

	double l2_kernel_sigmoid(int i, int j) const
	{
		return tanh(gamma2*(this->*lin_kernel_function)(i,j)+coef02);
	}
};
/*
struct smm_model *smm_train(const struct smm_problem *prob, const struct smm_parameter *param);
void smm_cross_validation(const struct smm_problem *prob, const struct smm_parameter *param, int nr_fold, double *target);

int smm_save_model(const char *model_file_name, const struct smm_model *model);
struct smm_model *smm_load_model(const char *model_file_name);

int smm_get_svm_type(const struct smm_model *model);
int smm_get_nr_class(const struct smm_model *model);
void smm_get_labels(const struct smm_model *model, int *label);
double smm_get_svr_probability(const struct smm_model *model);

double svm_predict_values(const struct smm_model *model, const struct smm_node *x, double* dec_values);
double svm_predict(const struct smm_model *model, const struct smm_node *x);
double svm_predict_probability(const struct smm_model *model, const struct smm_node *x, double* prob_estimates);

double smm_predict_values(const smm_model *model, const smm_node **x, int n, double* dec_values);
double smm_predict(const smm_model *model, const smm_node **x, int n);
double smm_predict_probability(const smm_model *model, const smm_node **x, int n, double *prob_estimates);

void smm_free_model_content(struct smm_model *model_ptr);
void smm_free_and_destroy_model(struct smm_model **model_ptr_ptr);
void smm_destroy_param(struct smm_parameter *param);

const char *smm_check_parameter(const struct smm_problem *prob, const struct smm_parameter *param);
int smm_check_probability_model(const struct smm_model *model);

void smm_set_print_string_function(void (*print_func)(const char *));

#ifdef __cplusplus
}
#endif

#endif /* _LIBSMM_H */