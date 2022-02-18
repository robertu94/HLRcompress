#ifndef __HLRCOMPRESS_BLAS_BLAS_DEF_HH
#define __HLRCOMPRESS_BLAS_BLAS_DEF_HH
//
// Project     : HLRcompress
// Module      : blas/blas_def
// Description : definition of BLAS/LAPACK functions in C-format
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hlrcompress/config.h>

// prevents issues with Windows build environment and dot-wrappers
#if USE_MKL == 1
#include <mkl_cblas.h>
#endif

#include <hlrcompress/misc/type_traits.hh>

namespace hlrcompress { namespace blas {

/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
//
// BLAS integer type
//
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

// define 32-bit integers vs. 64-bit integers
#if HLRCOMPRESS_USE_ILP64 == 1
using int_t = long;   // ILP64
#else
using int_t = int;    // LP64
#endif

// to query optimal workspace in LAPACK
constexpr int_t  LAPACK_WS_QUERY = -1;

/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
//
// declaration of external BLAS functions
//
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

// just in case we need extra definitions
#define  CFUNCDECL

extern "C" {

///////////////////////////////////////////////////////////////////
//
// real-valued functions (single precision)
//
///////////////////////////////////////////////////////////////////

// set (part of) A to alpha/beta
CFUNCDECL
void
slaset_  ( const char *  UPLO,
           const int_t * M,
           const int_t * N,
           const float * ALPHA,
           const float * BETA,
           float *       A,
           const int_t * LDA   );

// copy (part of) A to B
CFUNCDECL
void
slacpy_  ( const char *  UPLO,
           const int_t * M,
           const int_t * N,
           const float * A,
           const int_t * LDA,
           float *       B,
           const int_t * LDB  );

// compute dotproduct between x and y
CFUNCDECL
float
sdot_ ( const int_t * n,
        const float * dx,
        const int_t * incx,
        const float * dy,
        const int_t * incy);

// copy vector x into vector y
CFUNCDECL
void
scopy_ ( const int_t * n,
         const float * dx,
         const int_t * incx,
         float       * dy,
         const int_t * incy);

// compute y = y + a * x
CFUNCDECL
void
saxpy_ ( const int_t * n,
         const float * da,
         const float * dx,
         const int_t * incx,
         float *       dy,
         const int_t * incy );

// compute sum of absolut values of x
CFUNCDECL
float
sasum_  ( const int_t * n,
          const float * dx,
          const int_t * incx );

// return euclidean norm of x
CFUNCDECL
float
snrm2_  ( const int_t * n,
          const float * dx,
          const int_t * incx );

// scale x by a
CFUNCDECL
void
sscal_ ( const int_t * n,
         const float * da,
         float *       dx,
         const int_t * incx );

// interchange x and y
CFUNCDECL
void
sswap_ ( const int_t * n,
         float *       dx,
         const int_t * incx,
         float *       dy,
         const int_t * incy );

// finds index with element having maximal absolut value
CFUNCDECL
int_t
isamax_ ( const int_t * n,
          const float   *,
          const int_t * );

// y = alpha A x + beta y
CFUNCDECL
void
sgemv_ ( const char *  trans,
         const int_t * M,
         const int_t * N,
         const float * alpha,
         const float * A,
         const int_t * lda,
         const float * dx,
         const int_t * incx,
         const float * beta,
         float *       dy,
         const int_t * incy );

// compute c = alpha * a * b + beta * c
CFUNCDECL
void
sgemm_ ( const char *  transa,
         const char *  transb,
         const int_t * n,
         const int_t * m,
         const int_t * k,
         const float * alpha,
         const float * a,
         const int_t * lda,
         const float * b,
         const int_t * ldb,
         const float * beta,
         float *       c,
         const int_t * ldc );

// computes y = op(A)*x for upper/lower triangular A (y overwrites x)
int_t
strmv_ ( const char *  uplo,
         const char *  trans,
         const char *  diag,
         const int_t * n,
         const float * A,
         const int_t * ldA,
         float *       x,
         const int_t * incx );

// computes B = alpha * op(A) * B or B = alpha * B * op(A)
CFUNCDECL
void
strmm_ ( const char *           side,
         const char *           uplo,
         const char *           transa,
         const char *           diag,
         const int_t *     n,
         const int_t *     m,
         const float *          alpha,
         const float *          A,
         const int_t *     lda,
         float *                B,
         const int_t *     ldb );

// solves A x = b or A^T x = b with triangular A
CFUNCDECL
void
strsv_ ( const char *  uplo,
         const char *  trans,
         const char *  diag,
         const int_t * n,
         const float * A,
         const int_t * lda,
         float *       x,
         const int_t * incx );

// solves A X = B or A^T X = B with triangular A
CFUNCDECL
void
strsm_ ( const char *  side,
         const char *  uplo,
         const char *  trans,
         const char *  diag,
         const int_t * n,
         const int_t * m,
         const float * alpha,
         const float * A,
         const int_t * lda,
         float *       B,
         const int_t * ldb );

// A = alpha * x * y^T + A, A \in \R^{m x n}
CFUNCDECL
void
sger_ ( const int_t * m,
        const int_t * n,
        const float * alpha,
        const float * x,
        const int_t * incx,
        const float * y,
        const int_t * incy,
        float *       A,
        const int_t * lda );

///////////////////////////////////////////////////////////////////
//
// real-valued functions (double precision)
//
///////////////////////////////////////////////////////////////////

// set (part of) A to alpha/beta
CFUNCDECL
void
dlaset_  ( const char *   UPLO,
           const int_t *  M,
           const int_t *  N,
           const double * ALPHA,
           const double * BETA,
           double *       A,
           const int_t *  LDA   );

// copy (part of) A to B
CFUNCDECL
void
dlacpy_  ( const char *   UPLO,
           const int_t *  M,
           const int_t *  N,
           const double * A,
           const int_t *  LDA,
           double *       B,
           const int_t *  LDB  );

// compute dotproduct between x and y
CFUNCDECL
double
ddot_ ( const int_t *  n,
        const double * dx,
        const int_t *  incx,
        const double * dy,
        const int_t *  incy);
    
// copy vector x into vector y
CFUNCDECL
void
dcopy_ ( const int_t *  n,
         const double * dx,
         const int_t *  incx,
         double       * dy,
         const int_t *  incy);

// compute y = y + a * x
CFUNCDECL
void
daxpy_ ( const int_t *  n,
         const double * da,
         const double * dx,
         const int_t *  incx,
         double *       dy,
         const int_t *  incy );

// compute sum of absolut values of x
CFUNCDECL
double
dasum_  ( const int_t *  n,
          const double * dx,
          const int_t *  incx );

// return euclidean norm of x
CFUNCDECL
double
dnrm2_  ( const int_t *  n,
          const double * dx,
          const int_t *  incx );

// scale x by a
CFUNCDECL
void
dscal_ ( const int_t *  n,
         const double * da,
         double *       dx,
         const int_t *  incx );

// interchange x and y
CFUNCDECL
void
dswap_ ( const int_t * n,
         double *      dx,
         const int_t * incx,
         double *      dy,
         const int_t * incy );

// finds index with element having maximal absolut value
CFUNCDECL
int_t
idamax_ ( const int_t *    n,
          const double *,
          const int_t * );
    
// y = alpha A x + beta y
CFUNCDECL
void
dgemv_ ( const char *   trans,
         const int_t *  M,
         const int_t *  N,
         const double * alpha,
         const double * A,
         const int_t *  lda,
         const double * dx,
         const int_t *  incx,
         const double * beta,
         double *       dy,
         const int_t *  incy );

// compute c = alpha * a * b + beta * c
CFUNCDECL
void
dgemm_ ( const char *   transa,
         const char *   transb,
         const int_t *  n,
         const int_t *  m,
         const int_t *  k,
         const double * alpha,
         const double * a,
         const int_t *  lda,
         const double * b,
         const int_t *  ldb,
         const double * beta,
         double *       c,
         const int_t *  ldc );

// computes y = op(A)*x for upper/lower triangular A (y overwrites x)
void
dtrmv_ ( const char *   uplo,
         const char *   trans,
         const char *   diag,
         const int_t *  n,
         const double * A,
         const int_t *  ldA,
         double *       x,
         const int_t *  incx );

// computes B = alpha * op(A) * B or B = alpha * B * op(A)
CFUNCDECL
void
dtrmm_ ( const char *   side,
         const char *   uplo,
         const char *   transa,
         const char *   diag,
         const int_t *  n,
         const int_t *  m,
         const double * alpha,
         const double * A,
         const int_t *  lda,
         double *       B,
         const int_t *  ldb );

// solves A x = b or A^T x = b with triangular A
CFUNCDECL
void
dtrsv_ ( const char *   uplo,
         const char *   trans,
         const char *   diag,
         const int_t *  n,
         const double * A,
         const int_t *  lda,
         double *       x,
         const int_t *  incx );
    
// solves A X = B or A^T X = B with triangular A
CFUNCDECL
void
dtrsm_ ( const char *   side,
         const char *   uplo,
         const char *   trans,
         const char *   diag,
         const int_t *  n,
         const int_t *  m,
         const double * alpha,
         const double * A,
         const int_t *  lda,
         double *       B,
         const int_t *  ldb );

// A = alpha * x * y^T + A, A \in \R^{m x n}
CFUNCDECL
void
dger_ ( const int_t *  m,
        const int_t *  n,
        const double * alpha,
        const double * x,
        const int_t *  incx,
        const double * y,
        const int_t *  incy,
        double *       A,
        const int_t *  lda );

///////////////////////////////////////////////////////////////////
//
// complex-valued functions (single precision)
//
///////////////////////////////////////////////////////////////////

// set (part of) A to alpha/beta
CFUNCDECL
void
claset_  ( const char *                UPLO,
           const int_t *               M,
           const int_t *               N,
           const std::complex<float> * ALPHA,
           const std::complex<float> * BETA,
           std::complex<float> *       A,
           const int_t *               LDA   );

// copy (part of) A to B
CFUNCDECL
void
clacpy_  ( const char *                UPLO,
           const int_t *               M,
           const int_t *               N,
           const std::complex<float> * A,
           const int_t *               LDA,
           std::complex<float> *       B,
           const int_t *               LDB  );

// compute dotproduct between x and y
CFUNCDECL
void
xcdotu_ ( const int_t *               n,
          const std::complex<float> * dx,
          const int_t *               incx,
          const std::complex<float> * dy,
          const int_t *               incy,
          std::complex<float> *       retval );
CFUNCDECL
void
xcdotc_ ( const int_t *               n,
          const std::complex<float> * dx,
          const int_t *               incx,
          const std::complex<float> * dy,
          const int_t *               incy,
          std::complex<float> *       retval );

// copy vector x into vector y
CFUNCDECL
void
ccopy_ ( const int_t *               n,
         const std::complex<float> * dx,
         const int_t *               incx,
         std::complex<float> *       dy,
         const int_t *               incy);

// compute y = y + a * x
CFUNCDECL
void
caxpy_ ( const int_t *               n,
         const std::complex<float> * da,
         const std::complex<float> * dx,
         const int_t *               incx,
         std::complex<float> *       dy,
         const int_t *               incy );

// compute euclidean norm
CFUNCDECL
float
scnrm2_ ( const int_t *               n,
          const std::complex<float> * x,
          const int_t *               incx );

// scale x by a
CFUNCDECL
void
cscal_  ( const int_t *               n,
          const std::complex<float> * da,
          std::complex<float> *       dx,
          const int_t *               incx );
CFUNCDECL
void
csscal_ ( const int_t *         n,
          const float *         da,
          std::complex<float> * dx,
          const int_t *         incx );

// interchange x and y
CFUNCDECL
void
cswap_ ( const int_t *         n,
         std::complex<float> * dx,
         const int_t *         incx,
         std::complex<float> * dy,
         const int_t *         incy );

// finds index with element having maximal absolut value
CFUNCDECL
int_t
icamax_ ( const int_t * n,
          const std::complex<float> *,
          const int_t * );

// y = alpha A x + beta y
CFUNCDECL
void
cgemv_ ( const char *                trans,
         const int_t *               M,
         const int_t *               N,
         const std::complex<float> * alpha,
         const std::complex<float> * A,
         const int_t *               lda,
         const std::complex<float> * dx,
         const int_t *               incx,
         const std::complex<float> * beta,
         std::complex<float> *       dy,
         const int_t *               incy );

// compute c = alpha * a * b + beta * c
CFUNCDECL
void
cgemm_ ( const char *                transa,
         const char *                transb,
         const int_t *               n,
         const int_t *               m,
         const int_t *               k,
         const std::complex<float> * alpha,
         const std::complex<float> * a,
         const int_t *               lda,
         const std::complex<float> * b,
         const int_t *               ldb,
         const std::complex<float> * beta,
         std::complex<float> *       c,
         const int_t *               ldc );

// computes y = op(A)*x for upper/lower triangular A (y overwrites x)
void
ctrmv_ ( const char *                uplo,
         const char *                trans,
         const char *                diag,
         const int_t *               n,
         const std::complex<float> * A,
         const int_t *               ldA,
         std::complex<float> *       x,
         const int_t *               incx );

// computes B = alpha * op(A) * B or B = alpha * B * op(A)
CFUNCDECL
void
ctrmm_ ( const char *                side,
         const char *                uplo,
         const char *                transa,
         const char *                diag,
         const int_t *               m,
         const int_t *               n,
         const std::complex<float> * alpha,
         const std::complex<float> * a,
         const int_t *               lda,
         std::complex<float> *       b,
         const int_t *               ldb );

// solves A x = b or A^T x = b with triangular A
CFUNCDECL
void
ctrsv_ ( const char *                uplo,
         const char *                trans,
         const char *                diag,
         const int_t *               n,
         const std::complex<float> * a,
         const int_t *               lda,
         std::complex<float> *       x,
         const int_t *               incx );

// solves A X = B or A^T X = B with triangular A
CFUNCDECL
void
ctrsm_ ( const char *                  side,
         const char *                  uplo,
         const char *                  trans,
         const char *                  diag,
         const int_t *                 n,
         const int_t *                 m,
         const std::complex< float > * alpha,
         const std::complex< float > * A,
         const int_t *                 lda,
         std::complex< float > *       B,
         const int_t *                 ldb );

// rank-1 update: A = alpha * x * y^T + A
CFUNCDECL
void
cgeru_ ( const int_t *               m,
         const int_t *               n,
         const std::complex<float> * alpha,
         const std::complex<float> * x,
         const int_t *               incx,
         const std::complex<float> * y,
         const int_t *               incy,
         std::complex<float> *       A,
         const int_t *               lda );

// rank-1 update: A = alpha * x * conj(y^T) + A
CFUNCDECL
void
cgerc_ ( const int_t *               m,
         const int_t *               n,
         const std::complex<float> * alpha,
         const std::complex<float> * x,
         const int_t *               incx,
         const std::complex<float> * y,
         const int_t *               incy,
         std::complex<float> *       A,
         const int_t *               lda );

///////////////////////////////////////////////////////////////////
//
// complex-valued functions (double precision)
//
///////////////////////////////////////////////////////////////////

// set (part of) A to alpha/beta
CFUNCDECL
void
zlaset_  ( const char *                 UPLO,
           const int_t *                M,
           const int_t *                N,
           const std::complex<double> * ALPHA,
           const std::complex<double> * BETA,
           std::complex<double> *       A,
           const int_t *                LDA   );

// copy (part of) A to B
CFUNCDECL
void
zlacpy_  ( const char *                 UPLO,
           const int_t *                M,
           const int_t *                N,
           const std::complex<double> * A,
           const int_t *                LDA,
           std::complex<double> *       B,
           const int_t *                LDB  );

// compute dotproduct between x and y
CFUNCDECL
void
xzdotu_ ( const int_t *                n,
          const std::complex<double> * dx,
          const int_t *                incx,
          const std::complex<double> * dy,
          const int_t *                incy,
          std::complex<double> *       retval );
CFUNCDECL
void
xzdotc_ ( const int_t *                n,
          const std::complex<double> * dx,
          const int_t *                incx,
          const std::complex<double> * dy,
          const int_t *                incy,
          std::complex<double> *       retval );

// copy vector x into vector y
CFUNCDECL
void
zcopy_ ( const int_t *                n,
         const std::complex<double> * dx,
         const int_t *                incx,
         std::complex<double>       * dy,
         const int_t *                incy);

// compute y = y + a * x
CFUNCDECL
void
zaxpy_ ( const int_t *                n,
         const std::complex<double> * da,
         const std::complex<double> * dx,
         const int_t *                incx,
         std::complex<double> *       dy,
         const int_t *                incy );

// compute euclidean norm
CFUNCDECL
double
dznrm2_ ( const int_t *                n,
          const std::complex<double> * x,
          const int_t *                incx );

// scale x by a
CFUNCDECL
void
zscal_  ( const int_t *                n,
          const std::complex<double> * da,
          std::complex<double> *       dx,
          const int_t *                incx );
CFUNCDECL
void
zdscal_ ( const int_t *                n,
          const double *               da,
          std::complex<double> *       dx,
          const int_t *                incx );

// interchange x and y
CFUNCDECL
void
zswap_ ( const int_t *          n,
         std::complex<double> * dx,
         const int_t *          incx,
         std::complex<double> * dy,
         const int_t *          incy );

// finds index with element having maximal absolut value
CFUNCDECL
int_t
izamax_ ( const int_t * n,
          const std::complex<double> *,
          const int_t * );
    
// y = alpha A x + beta y
CFUNCDECL
void
zgemv_ ( const char *                 trans,
         const int_t *                M,
         const int_t *                N,
         const std::complex<double> * alpha,
         const std::complex<double> * A,
         const int_t *                lda,
         const std::complex<double> * dx,
         const int_t *                incx,
         const std::complex<double> * beta,
         std::complex<double> *       dy,
         const int_t *                incy );

// compute c = alpha * a * b + beta * c
CFUNCDECL
void
zgemm_ ( const char *                 transa,
         const char *                 transb,
         const int_t *                n,
         const int_t *                m,
         const int_t *                k,
         const std::complex<double> * alpha,
         const std::complex<double> * a,
         const int_t *                lda,
         const std::complex<double> * b,
         const int_t *                ldb,
         const std::complex<double> * beta,
         std::complex<double> *       c,
         const int_t *                ldc );

// computes y = op(A)*x for upper/lower triangular A (y overwrites x)
void
ztrmv_ ( const char *                 uplo,
         const char *                 trans,
         const char *                 diag,
         const int_t *                n,
         const std::complex<double> * A,
         const int_t *                ldA,
         std::complex<double> *       x,
         const int_t *                incx );

// computes B = alpha * op(A) * B or B = alpha * B * op(A)
CFUNCDECL
void
ztrmm_ ( const char *                 side,
         const char *                 uplo,
         const char *                 transa,
         const char *                 diag,
         const int_t *                m,
         const int_t *                n,
         const std::complex<double> * alpha,
         const std::complex<double> * a,
         const int_t *                lda,
         std::complex<double> *       b,
         const int_t *                ldb );

// solves A x = b or A^T x = b with triangular A
CFUNCDECL
void
ztrsv_ ( const char *                 uplo,
         const char *                 trans,
         const char *                 diag,
         const int_t *                n,
         const std::complex<double> * a,
         const int_t *                lda,
         std::complex<double> *       x,
         const int_t *                incx );

// solves A X = B or A^T X = B with triangular A
CFUNCDECL
void
ztrsm_ ( const char *                   side,
         const char *                   uplo,
         const char *                   trans,
         const char *                   diag,
         const int_t *                  n,
         const int_t *                  m,
         const std::complex< double > * alpha,
         const std::complex< double > * A,
         const int_t *                  lda,
         std::complex< double > *       B,
         const int_t *                  ldb );

// rank-1 update: A = alpha * x * y^T + A
CFUNCDECL
void
zgeru_ ( const int_t *                m,
         const int_t *                n,
         const std::complex<double> * alpha,
         const std::complex<double> * x,
         const int_t *                incx,
         const std::complex<double> * y,
         const int_t *                incy,
         std::complex<double> *       A,
         const int_t *                lda );

// rank-1 update: A = alpha * x * conj(y^T) + A
CFUNCDECL
void
zgerc_ ( const int_t *                m,
         const int_t *                n,
         const std::complex<double> * alpha,
         const std::complex<double> * x,
         const int_t *                incx,
         const std::complex<double> * y,
         const int_t *                incy,
         std::complex<double> *       A,
         const int_t *                lda );
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//
// definition of external Lapack functions
//
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

extern "C" {
    
//////////////////////////////////////////////////////////////
//
// real-valued functions (single precision)
//
//////////////////////////////////////////////////////////////

// solve linear system of equations
CFUNCDECL
void
sgesv_   ( const int_t *   n,
           const int_t *   nrhs,
           float *         A,
           const int_t *   lda,
           int_t *         ipiv,
           float *         B,
           const int_t *   ldb,
           int_t *         info );

// compute inverse of triangular system
CFUNCDECL
void
strtri_  ( const char *    uplo,
           const char *    diag,
           const int_t *   n,
           float *         A,
           const int_t *   lda,
           int_t *         info );

// compute eigenvalues and eigenvectors of a tridiagonal, symmetric matrix
CFUNCDECL
void
sstev_   ( const char *    jobz,
           const int_t *   n,
           float *         D,
           float *         E,
           float *         Z,
           const int_t *   ldz,
           float *         work,
           int_t *         info );

// compute eigenvalues and eigenvectors of a symmetric matrix
CFUNCDECL
void
ssyev_   ( const char *    jobz,
           const char *    uplo,
           const int_t *   n,
           float *         A,
           const int_t *   lda,
           float *         w,
           float *         work,
           const int_t *   lwork,
           int_t *         info );
    
// compute selected eigenvalues and eigenvectors of a symmetric matrix
CFUNCDECL
void ssyevx_ ( const char * jobz, const char * range, const char * uplo,
               const int_t * n, float * A, const int_t * ldA,
               const float * vl, const float * vu, const int_t * il, const int_t * iu,
               const float * abstol, int_t * m, float * W, float * Z, const int_t * ldZ,
               float * work, const int_t * lwork, int_t * iwork, int_t * ifail,
               int_t * info );
    
// compute singular-value-decomposition
CFUNCDECL
void
sgesvd_  ( const char *    jobu,
           const char *    jobv,
           const int_t *   n,
           const int_t *   m,
           float *         A,
           const int_t *   lda,
           float *         S,
           float *         U,
           const int_t *   ldu,
           float *         V,
           const int_t *   ldv,
           float *         work,
           const int_t *   lwork,
           int_t *         info );

CFUNCDECL
void
sgesdd_  ( const char *    job,
           const int_t *   n,
           const int_t *   m,
           float *         A,
           const int_t *   lda,
           float *         S,
           float *         U,
           const int_t *   ldu,
           float *         VT,
           const int_t *   ldvt,
           float *         work,
           const int_t *   lwork,
           int_t *         iwork,
           int_t *         info );

CFUNCDECL
void
sgesvj_  ( const char *    joba,
           const char *    jobu,
           const char *    jobv,
           const int_t *   m,
           const int_t *   n,
           float *         a,
           const int_t *   lda,
           float *         sva,
           const int_t *   mv,
           float *         v,
           const int_t *   ldv,
           float *         work,
           const int_t *   lwork,
           int_t *         info );

CFUNCDECL
void
sgejsv_  ( const char *    joba,
           const char *    jobu,
           const char *    jobv,
           const char *    jobr,
           const char *    jobt,
           const char *    jobp,
           const int_t *   m,
           const int_t *   n,
           float *         a,
           const int_t *   lda,
           float *         sva,
           float *         u,
           const int_t *   ldu,
           float *         v,
           const int_t *   ldv,
           float *         work,
           const int_t *   lwork,
           int_t *         iwork,
           int_t *         info );

// compute QR-factorisation
CFUNCDECL
void
sgeqrf_  ( const int_t *   m,
           const int_t *   n,
           float *         A,
           const int_t *   lda,
           float *         tau,
           float *         work,
           const int_t *   lwork,
           int_t *         info );

CFUNCDECL
void
sgeqr2_ ( const int_t *  nrows,
          const int_t *  ncols,
          float *        A,
          const int_t *  lda,
          float *        tau,
          float *        work,
          const int_t *  info );

CFUNCDECL
void
sorgqr_  ( const int_t *   m,
           const int_t *   n,
           const int_t *   k,
           float *         A,
           const int_t *   lda,
           const float *   tau,
           float *         work,
           const int_t *   lwork,
           int_t *         info );

CFUNCDECL
void
sorg2r_ ( const int_t *        nrows,
          const int_t *        ncols,
          const int_t *        nref,
          float *              A,
          const int_t *        lda,
          const float *        tau,
          float *              work,
          const int_t *        info );

CFUNCDECL
void
sgeqp3_  ( const int_t *   m,
           const int_t *   n,
           float *         A,
           const int_t *   lda,
           int_t *         jpvt,
           float *         tau,
           float *         work,
           const int_t *   lwork,
           int_t *         info );

// compute LU factorisation of given matrix A
CFUNCDECL
void
sgetrf_  ( const int_t *   m,
           const int_t *   n,
           float *         A,
           const int_t *   lda,
           int_t *         ipiv,
           int_t *         info );

// compute inverse of A (using result from getrf)
CFUNCDECL
void
sgetri_  ( const int_t *   n,
           float *         A,
           const int_t *   lda,
           int_t *         ipiv,
           float *         work,
           const int_t *   lwork,
           int_t *         info );

// determine machine parameters
CFUNCDECL
float
slamch_  ( char *          cmach );

// generate plane-rotation
CFUNCDECL
int_t
slartg_  ( float *         f,
           float *         g,
           float *         cs,
           float *         sn,
           float *         r );

// compute householder reflection
CFUNCDECL
void
slarfg_ ( const int_t *      n,
          const float *      alpha,
          const float *      x,
          const int_t *      incx,
          const float *      tau );

// apply householder reflection
CFUNCDECL
void
slarf_  ( const char *       side,
          const int_t *      n,
          const int_t *      m,
          const float *      V,
          const int_t *      incv,
          const float *      tau,
          float *            C,
          const int_t *      ldc,
          const float *      work );

//////////////////////////////////////////////////////////////
//
// real-valued functions (double precision)
//
//////////////////////////////////////////////////////////////

// solve linear system of equations
CFUNCDECL
void
dgesv_   ( const int_t *   n,
           const int_t *   nrhs,
           double *        A,
           const int_t *   lda,
           int_t *         ipiv,
           double *        B,
           const int_t *   ldb,
           int_t *         info );

// compute inverse of triangular system
CFUNCDECL
void
dtrtri_  ( const char *    uplo,
           const char *    diag,
           const int_t *   n,
           double *        A,
           const int_t *   lda,
           int_t *         info );
    
// compute eigenvalues and eigenvectors of a tridiagonal, symmetric matrix
CFUNCDECL
void
dstev_   ( const char *    jobz,
           const int_t *   n,
           double *        D,
           double *        E,
           double *        Z,
           const int_t *   ldz,
           double *        work,
           int_t *         info );
    
// compute eigenvalues and eigenvectors of a symmetric matrix
CFUNCDECL
void
dsyev_   ( const char *    jobz,
           const char *    uplo,
           const int_t *   n,
           double *        A,
           const int_t *   lda,
           double *        w,
           double *        work,
           const int_t *   lwork,
           int_t *         info );
    
// compute selected eigenvalues and eigenvectors of a symmetric matrix
CFUNCDECL
void
dsyevx_  ( const char *    jobz,
           const char *    range,
           const char *    uplo,
           const int_t *   n,
           double *        A,
           const int_t *   ldA,
           const double *  vl,
           const double *  vu,
           const int_t *   il,
           const int_t *   iu,
           const double *  abstol,
           int_t *         m,
           double *        W,
           double *        Z,
           const int_t *   ldZ,
           double *        work,
           const int_t *   lwork,
           int_t *         iwork,
           int_t *         ifail,
           int_t *         info );
    
// compute singular-value-decomposition
CFUNCDECL
void
dgesvd_  ( const char *    jobu,
           const char *    jobv,
           const int_t *   n,
           const int_t *   m,
           double *        A,
           const int_t *   lda,
           double *        S,
           double *        U,
           const int_t *   ldu,
           double *        V,
           const int_t *   ldv,
           double *        work,
           const int_t *   lwork,
           int_t *         info );

CFUNCDECL
void
dgesdd_  ( const char *    jobz,
           const int_t *   n,
           const int_t *   m,
           double *        A,
           const int_t *   lda,
           double *        S,
           double *        U,
           const int_t *   ldu,
           double *        VT,
           const int_t *   ldvt,
           double *        work,
           const int_t *   lwork,
           int_t *         iwork,
           int_t *         info );

CFUNCDECL
void
dgejsv_  ( const char *    joba,
           const char *    jobu,
           const char *    jobv,
           const char *    jobr,
           const char *    jobt,
           const char *    jobp,
           const int_t *   m,
           const int_t *   n,
           double *        a,
           const int_t *   lda,
           double *        sva,
           double *        u,
           const int_t *   ldu,
           double *        v,
           const int_t *   ldv,
           double *        work,
           const int_t *   lwork,
           int_t *         iwork,
           int_t *         info );

CFUNCDECL
void
dgesvj_  ( const char *    joba,
           const char *    jobu,
           const char *    jobv,
           const int_t *   m,
           const int_t *   n,
           double *        a,
           const int_t *   lda,
           double *        sva,
           const int_t *   mv,
           double *        v,
           const int_t *   ldv,
           double *        work,
           const int_t *   lwork,
           int_t *         info );

// compute QR-factorisation
CFUNCDECL
void
dgeqrf_  ( const int_t *   m,
           const int_t *   n,
           double *        A,
           const int_t *   lda,
           double *        tau,
           double *        work,
           const int_t *   lwork,
           int_t *         info );

CFUNCDECL
void
dgeqr2_ ( const int_t *  nrows,
          const int_t *  ncols,
          double *       A,
          const int_t *  lda,
          double *       tau,
          double *       work,
          const int_t *  info );

CFUNCDECL
void
dorgqr_  ( const int_t *   m,
           const int_t *   n,
           const int_t *   k,
           double *        A,
           const int_t *   lda,
           const double *  tau,
           double *        work,
           const int_t *   lwork,
           int_t *         info );

CFUNCDECL
void
dorg2r_ ( const int_t *        nrows,
          const int_t *        ncols,
          const int_t *        nref,
          double *             A,
          const int_t *        lda,
          const double *       tau,
          double *             work,
          const int_t *        info );

CFUNCDECL
void
dgeqp3_  ( const int_t *   m,
           const int_t *   n,
           double *        A,
           const int_t *   lda,
           int_t *         jpvt,
           double *        tau,
           double *        work,
           const int_t *   lwork,
           int_t *         info );

// compute LU factorisation of given matrix A
CFUNCDECL
void
dgetrf_  ( const int_t *   m,
           const int_t *   n,
           double *        A,
           const int_t *   lda,
           int_t *         ipiv,
           int_t *         info );
    
// compute inverse of A (using result from getrf)
CFUNCDECL
void
dgetri_  ( const int_t *   n,
           double *        A,
           const int_t *   lda,
           int_t *         ipiv,
           double *        work,
           const int_t *   lwork,
           int_t *         info );
    
// determine machine parameters
CFUNCDECL
double
dlamch_  ( char *            cmach );

// generate plane-rotation
CFUNCDECL
int_t
dlartg_  ( double *          f,
           double *          g,
           double *          cs,
           double *          sn,
           double *          r );

// compute householder reflection
CFUNCDECL
void
dlarfg_ ( const int_t *      n,
          const double *     alpha,
          const double *     x,
          const int_t *      incx,
          const double *     tau );

// apply householder reflection
CFUNCDECL
void
dlarf_  ( const char *       side,
          const int_t *      n,
          const int_t *      m,
          const double *     V,
          const int_t *      incv,
          const double *     tau,
          double *           C,
          const int_t *      ldc,
          const double *     work );

//////////////////////////////////////////////////////////////
//
// complex-valued functions (single precision)
//
//////////////////////////////////////////////////////////////

// solve linear system of equations
CFUNCDECL
void
cgesv_   ( const int_t *           n,
           const int_t *           nrhs,
           std::complex<float> *   a,
           const int_t *           lda,
           int_t *                 ipiv,
           std::complex<float> *   b,
           const int_t *           ldb,
           int_t *                 info );

// compute inverse of triangular system
CFUNCDECL
void
ctrtri_  ( const char *             uplo,
           const char *             diag,
           const int_t *            n,
           std::complex< float > *  A,
           const int_t *            lda,
           int_t *                  info );
    
// compute eigenvalues and eigenvectors of a symmetric matrix
CFUNCDECL
void
cheev_   ( const char *            jobz,
           const char *            uplo,
           const int_t *           n,
           std::complex<float> *   A,
           const int_t *           lda,
           float *                 w,
           std::complex<float> *   work,
           const int_t *           lwork,
           float *                 rwork,
           int_t *                 info );
    
// compute singular-value-decomposition
CFUNCDECL
void
cgesvd_  ( const char *            jobu,
           const char *            jobv,
           const int_t *           n,
           const int_t *           m,
           std::complex<float> *   A,
           const int_t *           lda,
           float *                 S,
           std::complex<float> *   U,
           const int_t *           ldu,
           std::complex<float> *   V,
           const int_t *           ldv,
           std::complex<float> *   work,
           const int_t *           lwork,
           float *                 rwork,
           int_t *                 info );

CFUNCDECL
void
cgesdd_  ( const char *            job,
           const int_t *           n,
           const int_t *           m,
           std::complex<float> *   A,
           const int_t *           lda,
           float *                 S,
           std::complex<float> *   U,
           const int_t *           ldu,
           std::complex<float> *   VT,
           const int_t *           ldvt,
           std::complex<float> *   work,
           const int_t *           lwork,
           float *                 rwork,
           const int_t *           iwork,
           int_t *                 info );

CFUNCDECL
void
cgesvj_  ( const char *            joba,
           const char *            jobu,
           const char *            jobv,
           const int_t *           n,
           const int_t *           m,
           std::complex<float> *   A,
           const int_t *           lda,
           float *                 S,
           const int_t *           mv,
           std::complex<float> *   V,
           const int_t *           ldv,
           std::complex<float> *   cwork,
           const int_t *           lwork,
           float *                 rwork,
           const int_t *           lrwork,
           int_t *                 info );

// compute QR-factorisation
CFUNCDECL
void
cgeqrf_  ( const int_t *           m,
           const int_t *           n,
           std::complex<float> *   A,
           const int_t *           lda,
           std::complex<float> *   tau,
           std::complex<float> *   work,
           const int_t *           lwork,
           int_t *                 info );

CFUNCDECL
void
cgeqr2_ ( const int_t *             nrows,
          const int_t *             ncols,
          std::complex< float > *   A,
          const int_t *             lda,
          std::complex< float > *   tau,
          std::complex< float > *   work,
          const int_t *             info );

CFUNCDECL
void
cungqr_  ( const int_t *                m,
           const int_t *                n,
           const int_t *                k,
           std::complex<float> *        a,
           const int_t *                lda,
           const std::complex<float> *  tau,
           std::complex<float> *        work,
           const int_t *                lwork,
           int_t *                      info );

CFUNCDECL
void
cung2r_ ( const int_t *                 nrows,
          const int_t *                 ncols,
          const int_t *                 nref,
          std::complex< float > *       A,
          const int_t *                 lda,
          const std::complex< float > * tau,
          std::complex< float > *       work,
          const int_t *                 info );

CFUNCDECL
void
cgeqp3_  ( const int_t *          m,
           const int_t *          n,
           std::complex<float> *  A,
           const int_t *          lda,
           int_t *                jpvt,
           std::complex<float> *  tau,
           std::complex<float> *  work,
           const int_t *          lwork,
           float *                rwork,
           int_t *                info );

#if HAS_GEQP3_TRUNC == 1

CFUNCDECL
void
cgeqp3trunc_  ( const int_t *          m,
                const int_t *          n,
                std::complex<float> *  A,
                const int_t *          lda,
                int_t *                jpvt,
                std::complex<float> *  tau,
                int_t *                ntrunc,
                float *                atrunc,
                float *                rtrunc,
                std::complex<float> *  work,
                const int_t *          lwork,
                float *                rwork,
                int_t *                info );

#endif

// compute LU factorisation of given matrix A
CFUNCDECL
void
cgetrf_  ( const int_t *           m,
           const int_t *           n,
           std::complex<float> *   a,
           const int_t *           lda,
           int_t *                 ipiv,
           int_t *                 info);

// compute inverse of A (using result from getrf)
CFUNCDECL
void
cgetri_  ( const int_t *           n,
           std::complex<float> *   a,
           const int_t *           lda,
           int_t *                 ipiv,
           std::complex<float> *   work,
           const int_t *           lwork,
           int_t *                 info);

// compute householder reflection
CFUNCDECL
void
clarfg_ ( const int_t *                 n,
          const std::complex< float > * alpha,
          const std::complex< float > * x,
          const int_t *                 incx,
          const std::complex< float > * tau );

// apply householder reflection
CFUNCDECL
void
clarf_  ( const char *                  side,
          const int_t *                 n,
          const int_t *                 m,
          const std::complex< float > * V,
          const int_t *                 incv,
          const std::complex< float > * tau,
          std::complex< float > *       C,
          const int_t *                 ldc,
          const std::complex< float > * work );

//////////////////////////////////////////////////////////////
//
// complex-valued functions (double precision)
//
//////////////////////////////////////////////////////////////

// solve linear system of equations
CFUNCDECL
void
zgesv_   ( const int_t *           n,
           const int_t *           nrhs,
           std::complex<double> *  a,
           const int_t *           lda,
           int_t *                 ipiv,
           std::complex<double> *  b,
           const int_t *           ldb,
           int_t *                 info );

// compute inverse of triangular system
CFUNCDECL
void
ztrtri_  ( const char *                 uplo,
           const char *                 diag,
           const int_t *                n,
           std::complex< double > *     A,
           const int_t *                lda,
           int_t *                      info );
    
// compute eigenvalues and eigenvectors of a symmetric matrix
CFUNCDECL
void
zheev_   ( const char *            jobz,
           const char *            uplo,
           const int_t *           n,
           std::complex<double> *  A,
           const int_t *           lda,
           double *                w,
           std::complex<double> *  work,
           const int_t *           lwork,
           double *                rwork,
           int_t *                 info );
    
// compute singular-value-decomposition
CFUNCDECL
void
zgesvd_  ( const char *            jobu,
           const char *            jobv,
           const int_t *           n,
           const int_t *           m,
           std::complex<double> *  A,
           const int_t *           lda,
           double *                S,
           std::complex<double> *  U,
           const int_t *           ldu,
           std::complex<double> *  V,
           const int_t *           ldv,
           std::complex<double> *  work,
           const int_t *           lwork,
           double *                rwork,
           int_t *                 info );

CFUNCDECL
void
zgesdd_  ( const char *            job,
           const int_t *           n,
           const int_t *           m,
           std::complex<double> *  A,
           const int_t *           lda,
           double *                S,
           std::complex<double> *  U,
           const int_t *           ldu,
           std::complex<double> *  VT,
           const int_t *           ldvt,
           std::complex<double> *  work,
           const int_t *           lwork,
           double *                rwork,
           const int_t *           iwork,
           int_t *                 info );

CFUNCDECL
void
zgesvj_  ( const char *            joba,
           const char *            jobu,
           const char *            jobv,
           const int_t *           n,
           const int_t *           m,
           std::complex<double> *  A,
           const int_t *           lda,
           double *                S,
           const int_t *           mv,
           std::complex<double> *  V,
           const int_t *           ldv,
           std::complex<double> *  cwork,
           const int_t *           lwork,
           double *                rwork,
           const int_t *           lrwork,
           int_t *                 info );

// compute QR-factorisation
CFUNCDECL
void
zgeqrf_  ( const int_t *           m,
           const int_t *           n,
           std::complex<double> *  A,
           const int_t *           lda,
           std::complex<double> *  tau,
           std::complex<double> *  work,
           const int_t *           lwork,
           int_t *                 info );

CFUNCDECL
void
zgeqr2_ ( const int_t *             nrows,
          const int_t *             ncols,
          std::complex< double > *  A,
          const int_t *             lda,
          std::complex< double > *  tau,
          std::complex< double > *  work,
          const int_t *             info );

CFUNCDECL
void
zungqr_  ( const int_t *                m,
           const int_t *                n,
           const int_t *                k,
           std::complex<double> *       a,
           const int_t *                lda,
           const std::complex<double> * tau,
           std::complex<double> *       work,
           const int_t *                lwork,
           int_t *                      info );

CFUNCDECL
void
zung2r_ ( const int_t *                   nrows,
          const int_t *                   ncols,
          const int_t *                   nref,
          std::complex< double > *        A,
          const int_t *                   lda,
          const std::complex< double > *  tau,
          std::complex< double > *        work,
          const int_t *                   info );

CFUNCDECL
void
zgeqp3_  ( const int_t *           m,
           const int_t *           n,
           std::complex<double> *  A,
           const int_t *           lda,
           int_t *                 jpvt,
           std::complex<double> *  tau,
           std::complex<double> *  work,
           const int_t *           lwork,
           double *                rwork,
           int_t *                 info );

#if HAS_GEQP3_TRUNC == 1

CFUNCDECL
void
zgeqp3trunc_  ( const int_t *           m,
                const int_t *           n,
                std::complex<double> *  A,
                const int_t *           lda,
                int_t *                 jpvt,
                std::complex<double> *  tau,
                int_t *                 ntrunc,
                double *                atrunc,
                double *                rtrunc,
                std::complex<double> *  work,
                const int_t *           lwork,
                double *                rwork,
                int_t *                 info );

#endif

// compute LU factorisation of given matrix A
CFUNCDECL
void
zgetrf_  ( const int_t *           m,
           const int_t *           n,
           std::complex<double> *  a,
           const int_t *           lda,
           int_t *                 ipiv,
           int_t *                 info);

// compute inverse of A (using result from getrf)
CFUNCDECL
void
zgetri_  ( const int_t *                n,
           std::complex<double> *       a,
           const int_t *                lda,
           int_t *                      ipiv,
           std::complex<double> *       work,
           const int_t *                lwork,
           int_t *                      info);

// compute householder reflection
CFUNCDECL
void
zlarfg_ ( const int_t *                   n,
          const std::complex< double > *  alpha,
          const std::complex< double > *  x,
          const int_t *                   incx,
          const std::complex< double > *  tau );

// apply householder reflection
CFUNCDECL
void
zlarf_  ( const char *                    side,
          const int_t *                   n,
          const int_t *                   m,
          const std::complex< double > *  V,
          const int_t *                   incv,
          const std::complex< double > *  tau,
          std::complex< double > *        C,
          const int_t *                   ldc,
          const std::complex< double > *  work );

//////////////////////////////////////////////////////////////
//
// misc. helpers
//
//////////////////////////////////////////////////////////////

// return problem dependent parameters
int_t
ilaenv_ ( const int_t *    ispec,
          const char *     name,
          const char *     opts,
          const int_t *    n1,
          const int_t *    n2,
          const int_t *    n3,
          const int_t *    n4 );

}// extern "C"

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//
// wrappers for BLAS functions
//
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

//
// *laset
//
#define HLRCOMPRESS_LASET_FUNC( type, func )                \
    inline void laset ( const char   uplo,                  \
                        const int_t  m,                     \
                        const int_t  n,                     \
                        const type   alpha,                 \
                        const type   beta,                  \
                        type *       A,                     \
                        const int_t  lda ) {                \
        func( & uplo, & m, & n, & alpha, & beta, A, & lda ); }

HLRCOMPRESS_LASET_FUNC( float,                  slaset_ )
HLRCOMPRESS_LASET_FUNC( double,                 dlaset_ )
HLRCOMPRESS_LASET_FUNC( std::complex< float >,  claset_ )
HLRCOMPRESS_LASET_FUNC( std::complex< double >, zlaset_ )

#undef HLRCOMPRESS_LASET_FUNC

//
// *lacpy
//
#define HLRCOMPRESS_LACPY_FUNC( type, func )            \
    inline void lacpy ( const char    uplo,             \
                        const int_t   m,                \
                        const int_t   n,                \
                        const type *  A,                \
                        const int_t   lda,              \
                        type *        B,                \
                        const int_t   ldb ) {           \
        func( & uplo, & m, & n, A, & lda, B, & ldb ); }

HLRCOMPRESS_LACPY_FUNC( float,             slacpy_ )
HLRCOMPRESS_LACPY_FUNC( double,            dlacpy_ )
HLRCOMPRESS_LACPY_FUNC( std::complex< float >,  clacpy_ )
HLRCOMPRESS_LACPY_FUNC( std::complex< double >, zlacpy_ )

#undef HLRCOMPRESS_LACPY_FUNC

//
// *scal
//
#define HLRCOMPRESS_SCAL_FUNC( type, func )     \
    inline void scal ( const int_t n,           \
                       const type  alpha,       \
                       type *      x,           \
                       const int_t incx )       \
    { func( & n, & alpha, x, & incx ); }

HLRCOMPRESS_SCAL_FUNC( float,                  sscal_ )
HLRCOMPRESS_SCAL_FUNC( double,                 dscal_ )
HLRCOMPRESS_SCAL_FUNC( std::complex< float >,  cscal_ )
HLRCOMPRESS_SCAL_FUNC( std::complex< double >, zscal_ )

//
// *copy
//
#define HLRCOMPRESS_COPY_FUNC( type, func )            \
    inline void copy ( const int_t n, const type * x,  \
                       const int_t incx,               \
                       type * y, const int_t incy )    \
    { func( & n, x, & incx, y, & incy ); }
                                                       
HLRCOMPRESS_COPY_FUNC( float,                  scopy_ )
HLRCOMPRESS_COPY_FUNC( double,                 dcopy_ )
HLRCOMPRESS_COPY_FUNC( std::complex< float >,  ccopy_ )
HLRCOMPRESS_COPY_FUNC( std::complex< double >, zcopy_ )

//
// *swap
//
#define HLRCOMPRESS_SWAP_FUNC( type, func )            \
    inline  void swap ( const int_t n, type * x,       \
                        const int_t incx,              \
                        type * y, const int_t incy )   \
    { func( & n, x, & incx, y, & incy ); }
                                                       
HLRCOMPRESS_SWAP_FUNC( float,                  sswap_ )
HLRCOMPRESS_SWAP_FUNC( double,                 dswap_ )
HLRCOMPRESS_SWAP_FUNC( std::complex< float >,  cswap_ )
HLRCOMPRESS_SWAP_FUNC( std::complex< double >, zswap_ )

//
// i*amax
//
#define HLRCOMPRESS_MAX_IDX_FUNC( type, func )            \
    inline int_t max_idx ( const int_t n,                 \
                                type * x, const int_t incx )   \
    { return func( & n, x, & incx ); }
                                                       
HLRCOMPRESS_MAX_IDX_FUNC( float,                  isamax_ )
HLRCOMPRESS_MAX_IDX_FUNC( double,                 idamax_ )
HLRCOMPRESS_MAX_IDX_FUNC( std::complex< float >,  icamax_ )
HLRCOMPRESS_MAX_IDX_FUNC( std::complex< double >, izamax_ )

//
// *axpy
//
#define HLRCOMPRESS_AXPY_FUNC( type, func, flops )          \
    inline void axpy ( const int_t n,                       \
                       const type alpha, const type * x,    \
                       const int_t incx,                    \
                       type * y, const int_t incy )         \
    {                                                       \
        func( & n, & alpha, x, & incx, y, & incy );         \
    }

HLRCOMPRESS_AXPY_FUNC( float,                  saxpy_, SAXPY )
HLRCOMPRESS_AXPY_FUNC( double,                 daxpy_, DAXPY )
HLRCOMPRESS_AXPY_FUNC( std::complex< float >,  caxpy_, CAXPY )
HLRCOMPRESS_AXPY_FUNC( std::complex< double >, zaxpy_, ZAXPY )

//
// *dot(c)
//
inline
float
dot ( const int_t  n,
      const float *  x, const int_t  incx,
      const float *  y, const int_t  incy )
{
    return sdot_( & n, x, & incx, y, & incy );
}

inline
double
dot ( const int_t  n,
      const double *  x, const int_t  incx,
      const double *  y, const int_t  incy )
{
    return ddot_( & n, x, & incx, y, & incy );
}

inline
std::complex< float >
dot ( const int_t  n,
      const std::complex< float > *  x, const int_t  incx,
      const std::complex< float > *  y, const int_t  incy )
{
    std::complex< float >  res;
    
    #if USE_MKL == 1

    cblas_cdotc_sub( n, x, incx, y, incy, & res );
    
    #else
    
    xcdotc_( & n, x, & incx, y, & incy, & res );
    
    #endif
    
    return res;
}

inline
std::complex< double >
dot ( const int_t  n,
      const std::complex< double > *  x, const int_t  incx,
      const std::complex< double > *  y, const int_t  incy )
{
    std::complex< double >  res;
    
    #if USE_MKL == 1

    cblas_zdotc_sub( n, x, incx, y, incy, & res );
    
    #else
    
    xzdotc_( & n, x, & incx, y, & incy, & res );
    
    #endif
    
    return res;
}

//
// *dotu
//
inline
float
dotu ( const int_t n,
       const float * x, const int_t incx,
       const float * y, const int_t incy )
{
    return dot( n, x, incx, y, incy );
}

inline
double
dotu ( const int_t n,
       const double * x, const int_t incx,
       const double * y, const int_t incy )
{
    return dot( n, x, incx, y, incy );
}

inline
std::complex< float >
dotu ( const int_t  n,
       const std::complex< float > *  x, const int_t  incx,
       const std::complex< float > *  y, const int_t  incy )
{
    std::complex< float >  res;
    
    #if USE_MKL == 1

    cblas_cdotu_sub( n, x, incx, y, incy, & res );
    
    #else
    
    xcdotu_( & n, x, & incx, y, & incy, & res );
    
    #endif
    
    return res;
}

inline
std::complex< double >
dotu ( const int_t  n,
       const std::complex< double > *  x, const int_t  incx,
       const std::complex< double > *  y, const int_t  incy )
{
    std::complex< double >  res;
    
    #if USE_MKL == 1

    cblas_zdotu_sub( n, x, incx, y, incy, & res );
    
    #else
    
    xzdotu_( & n, x, & incx, y, & incy, & res );
    
    #endif
    
    return res;
}

//
// *norm2
//
#define HLRCOMPRESS_NORM2_FUNC( type, func, flops )                     \
    inline real_type_t< type >                           \
    norm2 ( const int_t n, const type * x, const int_t incx ) \
    {                                                                   \
        return func( & n, x, & incx );                                  \
    }

HLRCOMPRESS_NORM2_FUNC( float,                  snrm2_,  SDOT )
HLRCOMPRESS_NORM2_FUNC( double,                 dnrm2_,  DDOT )
HLRCOMPRESS_NORM2_FUNC( std::complex< float >,  scnrm2_, CDOT )
HLRCOMPRESS_NORM2_FUNC( std::complex< double >, dznrm2_, ZDOT )

//
// *ger
//
#define HLRCOMPRESS_GER_FUNC( type, func, flops )                       \
    inline                                                              \
    void ger ( const int_t n, const int_t m, const type alpha, \
               const type * x, const int_t incx,                   \
               const type * y, const int_t incy,                   \
               type * A, const int_t ldA )                         \
    {                                                                   \
        func( & n, & m, & alpha, x, & incx, y, & incy, A, & ldA );      \
    }

HLRCOMPRESS_GER_FUNC( float,                  sger_,  SGER )
HLRCOMPRESS_GER_FUNC( double,                 dger_ , DGER )
HLRCOMPRESS_GER_FUNC( std::complex< float >,  cgerc_, CGER )
HLRCOMPRESS_GER_FUNC( std::complex< double >, zgerc_, ZGER )

//
// *geru
//
#define HLRCOMPRESS_GERU_FUNC( type, func, flops )                      \
    inline                                                              \
    void geru ( const int_t n, const int_t m, const type alpha, \
                const type * x, const int_t incx,                  \
                const type * y, const int_t incy,                  \
                type * A, const int_t ldA )                        \
    {                                                                   \
        func( & n, & m, & alpha, x, & incx, y, & incy, A, & ldA );      \
    }

HLRCOMPRESS_GERU_FUNC( float,                  sger_,  SGER )
HLRCOMPRESS_GERU_FUNC( double,                 dger_,  DGER )
HLRCOMPRESS_GERU_FUNC( std::complex< float >,  cgeru_, CGER )
HLRCOMPRESS_GERU_FUNC( std::complex< double >, zgeru_, ZGER )

//
// *gemv
//
#define HLRCOMPRESS_GEMV_FUNC( type, func, flops )                      \
    inline                                                              \
    void gemv ( const char trans, const int_t n, const int_t m, const type alpha, \
                const type * A, const int_t ldA,                   \
                const type * x, const int_t incx,                  \
                const type beta, type * y, const int_t incy ) {    \
        char  ltrans = trans;                                           \
        if ( ! is_complex_type< type >::value && ( trans == 'C' ) )     \
            ltrans = 'T';                                               \
        func( & ltrans, & n, & m, & alpha, A, & ldA, x, & incx, & beta, y, & incy ); }

HLRCOMPRESS_GEMV_FUNC( float,                  sgemv_, SGEMV )
HLRCOMPRESS_GEMV_FUNC( double,                 dgemv_, DGEMV )
HLRCOMPRESS_GEMV_FUNC( std::complex< float >,  cgemv_, CGEMV )
HLRCOMPRESS_GEMV_FUNC( std::complex< double >, zgemv_, ZGEMV )

//
// *trmv
//
#define HLRCOMPRESS_TRMV_FUNC( type, func, flops )                      \
    inline                                                              \
    void trmv ( const char uplo, const char trans, const char diag,     \
                const int_t n, const type * A, const int_t ldA, \
                type * x, const int_t incx ) {                     \
        char  ltrans = trans;                                           \
        if ( ! is_complex_type< type >::value && ( trans == 'C' ) )     \
            ltrans = 'T';                                               \
        func( & uplo, & ltrans, & diag, & n, A, & ldA, x, & incx ); }

HLRCOMPRESS_TRMV_FUNC( float,                  strmv_, STRMV )
HLRCOMPRESS_TRMV_FUNC( double,                 dtrmv_, DTRMV )
HLRCOMPRESS_TRMV_FUNC( std::complex< float >,  ctrmv_, CTRMV )
HLRCOMPRESS_TRMV_FUNC( std::complex< double >, ztrmv_, ZTRMV )

//
// *trsv
//
#define HLRCOMPRESS_TRSV_FUNC( type, func, flops )                      \
    inline                                                              \
    void trsv ( const char uplo, const char trans, const char diag,     \
                const int_t n, const type * A, const int_t ldA, \
                type * b, const int_t incb ) {                     \
        char  ltrans = trans;                                           \
        if ( ! is_complex_type< type >::value && ( trans == 'C' ) )     \
            ltrans = 'T';                                               \
        func( & uplo, & ltrans, & diag, & n, A, & ldA, b, & incb ); }

HLRCOMPRESS_TRSV_FUNC( float,                  strsv_, STRSV )
HLRCOMPRESS_TRSV_FUNC( double,                 dtrsv_, DTRSV )
HLRCOMPRESS_TRSV_FUNC( std::complex< float >,  ctrsv_, CTRSV )
HLRCOMPRESS_TRSV_FUNC( std::complex< double >, ztrsv_, ZTRSV )

//
// *gemm
//
#define HLRCOMPRESS_GEMM_FUNC( type, func, flops )                      \
    inline                                                              \
    void gemm ( const char transA, const char transB,                   \
                const int_t n, const int_t m, const int_t k, \
                const type alpha,                                       \
                const type * A, const int_t ldA,                   \
                const type * B, const int_t ldB,                   \
                const type beta, type * C, const int_t ldC ) {     \
        char  ltransA = transA;                                         \
        char  ltransB = transB;                                         \
        if ( ! is_complex_type< type >::value && ( transA == 'C' ) )    \
            ltransA = 'T';                                              \
        if ( ! is_complex_type< type >::value && ( transB == 'C' ) )    \
            ltransB = 'T';                                              \
        func( & ltransA, & ltransB, & n, & m, & k,                      \
              & alpha, A, & ldA, B, & ldB,                              \
              & beta, C, & ldC ); }

HLRCOMPRESS_GEMM_FUNC( float,                  sgemm_, SGEMM )
HLRCOMPRESS_GEMM_FUNC( double,                 dgemm_, DGEMM )
HLRCOMPRESS_GEMM_FUNC( std::complex< float >,  cgemm_, CGEMM )
HLRCOMPRESS_GEMM_FUNC( std::complex< double >, zgemm_, ZGEMM )

//
// *trmm
//
#define HLRCOMPRESS_TRMM_FUNC( type, func, flops )                      \
    inline                                                              \
    void trmm ( const char side, const char uplo, const char trans, const char diag, \
                const int_t n, const int_t m, const type alpha, const type * A, \
                const int_t ldA, type * B, const int_t ldB ) { \
        char  ltrans = trans;                                           \
        if ( ! is_complex_type< type >::value && ( trans == 'C' ) )     \
            ltrans = 'T';                                               \
        func( & side, & uplo, & ltrans, & diag, & n, & m, & alpha, A, & ldA, B, & ldB ); }

HLRCOMPRESS_TRMM_FUNC( float,                  strmm_, STRMM )
HLRCOMPRESS_TRMM_FUNC( double,                 dtrmm_, CTRMM )
HLRCOMPRESS_TRMM_FUNC( std::complex< float >,  ctrmm_, DTRMM )
HLRCOMPRESS_TRMM_FUNC( std::complex< double >, ztrmm_, ZTRMM )

//
// *trsm
//
#define HLRCOMPRESS_TRSM_FUNC( type, func, flops )                      \
    inline                                                              \
    void trsm ( const char side, const char uplo, const char trans, const char diag, \
                const int_t n, const int_t m, const type alpha, const type * A, \
                const int_t ldA, type * B, const int_t ldB ) { \
        char  ltrans = trans;                                           \
        if ( ! is_complex_type< type >::value && ( trans == 'C' ) )     \
            ltrans = 'T';                                               \
        func( & side, & uplo, & ltrans, & diag, & n, & m, & alpha, A, & ldA, B, & ldB ); }

HLRCOMPRESS_TRSM_FUNC( float,                  strsm_, STRSM )
HLRCOMPRESS_TRSM_FUNC( double,                 dtrsm_, CTRSM )
HLRCOMPRESS_TRSM_FUNC( std::complex< float >,  ctrsm_, DTRSM )
HLRCOMPRESS_TRSM_FUNC( std::complex< double >, ztrsm_, ZTRSM )



//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//
// wrappers for LAPACK functions
//
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

//
// *gesv
//
#define HLRCOMPRESS_GESV_FUNC( type, func )                    \
    inline void gesv ( const int_t  n,                         \
                       const int_t  nrhs,                      \
                       type *       A,                         \
                       const int_t  ldA,                       \
                       int_t *      ipiv,                      \
                       type *       B,                         \
                       const int_t  ldB,                       \
                       int_t &      info ) {                   \
        info = 0;                                              \
        func( & n, & nrhs, A, & ldA, ipiv, B, & ldB, & info ); }

HLRCOMPRESS_GESV_FUNC( float,                  sgesv_ )
HLRCOMPRESS_GESV_FUNC( double,                 dgesv_ )
HLRCOMPRESS_GESV_FUNC( std::complex< float >,  cgesv_ )
HLRCOMPRESS_GESV_FUNC( std::complex< double >, zgesv_ )

#undef HLRCOMPRESS_GESV_FUNC

//
// *trtri
//
#define HLRCOMPRESS_TRTRI_FUNC( type, func )           \
    inline  void trtri ( const char   uplo,            \
                         const char   diag,            \
                         const int_t  n,               \
                         type *       A,               \
                         const int_t  ldA,             \
                         int_t &      info ) {         \
        info = 0;                                      \
        func( & uplo, & diag, & n, A, & ldA, & info ); }

HLRCOMPRESS_TRTRI_FUNC( float,                  strtri_ )
HLRCOMPRESS_TRTRI_FUNC( double,                 dtrtri_ )
HLRCOMPRESS_TRTRI_FUNC( std::complex< float >,  ctrtri_ )
HLRCOMPRESS_TRTRI_FUNC( std::complex< double >, ztrtri_ )

#undef HLRCOMPRESS_TRTRI_FUNC

//
// *getrf
//
#define HLRCOMPRESS_GETRF_FUNC( type, func, flops ) \
    inline void getrf ( const int_t  n,             \
                        const int_t  m,             \
                        type *       A,             \
                        const int_t  ldA,           \
                        int_t *      ipiv,          \
                        int_t &      info ) {       \
        info = 0;                                   \
        func( & n, & m, A, & ldA, ipiv, & info ); }

HLRCOMPRESS_GETRF_FUNC( float,                  sgetrf_, SGETRF )
HLRCOMPRESS_GETRF_FUNC( double,                 dgetrf_, CGETRF )
HLRCOMPRESS_GETRF_FUNC( std::complex< float >,  cgetrf_, DGETRF )
HLRCOMPRESS_GETRF_FUNC( std::complex< double >, zgetrf_, ZGETRF )

#undef HLRCOMPRESS_GETRF_FUNC

//
// *getri
//
#define HLRCOMPRESS_GETRI_FUNC( type, func, flops )        \
    inline void getri ( const int_t  n,                    \
                        type *       A,                    \
                        const int_t  ldA,                  \
                        int_t *      ipiv,                 \
                        type *       work,                 \
                        int_t        lwork,                \
                        int_t &      info ) {              \
        info = 0;                                          \
        func( & n, A, & ldA, ipiv, work, & lwork, & info ); }

HLRCOMPRESS_GETRI_FUNC( float,                  sgetri_, SGETRI )
HLRCOMPRESS_GETRI_FUNC( double,                 dgetri_, CGETRI )
HLRCOMPRESS_GETRI_FUNC( std::complex< float >,  cgetri_, DGETRI )
HLRCOMPRESS_GETRI_FUNC( std::complex< double >, zgetri_, ZGETRI )

#undef HLRCOMPRESS_GETRI_FUNC

//
// *geqrf
//
#define HLRCOMPRESS_GEQRF_FUNC( type, func, flops )            \
    inline void geqrf ( const int_t  n,                        \
                        const int_t  m,                        \
                        type *       A,                        \
                        const int_t  ldA,                      \
                        type *       tau,                      \
                        type *       work,                     \
                        int_t        lwork,                    \
                        int_t &      info ) {                  \
        info = 0;                                              \
        func( & n, & m, A, & ldA, tau, work, & lwork, & info ); }

HLRCOMPRESS_GEQRF_FUNC( float,                  sgeqrf_, SGEQRF )
HLRCOMPRESS_GEQRF_FUNC( double,                 dgeqrf_, CGEQRF )
HLRCOMPRESS_GEQRF_FUNC( std::complex< float >,  cgeqrf_, DGEQRF )
HLRCOMPRESS_GEQRF_FUNC( std::complex< double >, zgeqrf_, ZGEQRF )

#undef HLRCOMPRESS_GEQRF_FUNC

//
// *geqr2
//
#define HLRCOMPRESS_GEQR2_FUNC( type, func, flops )    \
    inline void geqr2 ( const int_t  n,                \
                        const int_t  m,                \
                        type *       A,                \
                        const int_t  ldA,              \
                        type *       tau,              \
                        type *       work,             \
                        int_t &      info ) {          \
        info = 0;                                      \
        func( & n, & m, A, & ldA, tau, work, & info ); }

HLRCOMPRESS_GEQR2_FUNC( float,                  sgeqr2_, SGEQRF )
HLRCOMPRESS_GEQR2_FUNC( double,                 dgeqr2_, CGEQRF )
HLRCOMPRESS_GEQR2_FUNC( std::complex< float >,  cgeqr2_, DGEQRF )
HLRCOMPRESS_GEQR2_FUNC( std::complex< double >, zgeqr2_, ZGEQRF )

#undef HLRCOMPRESS_GEQR2_FUNC

//
// *orgqr
//
#define HLRCOMPRESS_ORGQR_FUNC( type, func, flops )                \
    inline void orgqr ( const int_t  n,                            \
                        const int_t  m,                            \
                        const int_t  k,                            \
                        type *       A,                            \
                        const int_t  ldA,                          \
                        const type * tau,                          \
                        type *       work,                         \
                        int_t        lwork,                        \
                        int_t &      info ) {                      \
        info = 0;                                                  \
        func( & n, & m, & k, A, & ldA, tau, work, & lwork, & info ); }

HLRCOMPRESS_ORGQR_FUNC( float,             sorgqr_, SORGQR )
HLRCOMPRESS_ORGQR_FUNC( double,            dorgqr_, DORGQR )
HLRCOMPRESS_ORGQR_FUNC( std::complex< float >,  cungqr_, CUNGQR )
HLRCOMPRESS_ORGQR_FUNC( std::complex< double >, zungqr_, ZUNGQR )

#undef HLRCOMPRESS_ORGQR_FUNC

//
// *org2r
//
#define HLRCOMPRESS_ORG2R_FUNC( type, func, flops )        \
    inline void ung2r ( const int_t  n,                    \
                        const int_t  m,                    \
                        const int_t  k,                    \
                        type *       A,                    \
                        const int_t  ldA,                  \
                        const type * tau,                  \
                        type *       work,                 \
                        int_t &      info ) {              \
        info = 0;                                          \
        func( & n, & m, & k, A, & ldA, tau, work, & info ); }

HLRCOMPRESS_ORG2R_FUNC( float,                  sorg2r_, SORG2R )
HLRCOMPRESS_ORG2R_FUNC( double,                 dorg2r_, DORG2R )
HLRCOMPRESS_ORG2R_FUNC( std::complex< float >,  cung2r_, CUNG2R )
HLRCOMPRESS_ORG2R_FUNC( std::complex< double >, zung2r_, ZUNG2R )

#undef HLRCOMPRESS_ORG2R_FUNC

//
// *geqp3
//
#define HLRCOMPRESS_GEQP3_FUNC( type, func )                       \
    inline void geqp3 ( const int_t  n,                            \
                        const int_t  m,                            \
                        type *       A,                            \
                        const int_t  ldA,                          \
                        int_t *      jpvt,                         \
                        type *       tau,                          \
                        type *       work,                         \
                        int_t        lwork,                        \
                        real_type_t< type > *,                     \
                        int_t &      info ) {                      \
        info = 0;                                                       \
        func( & n, & m, A, & ldA, jpvt, tau, work, & lwork, & info ); }

HLRCOMPRESS_GEQP3_FUNC( float,  sgeqp3_ )
HLRCOMPRESS_GEQP3_FUNC( double, dgeqp3_ )

#undef HLRCOMPRESS_GEQP3_FUNC

#define HLRCOMPRESS_GEQP3_FUNC( type, func )                            \
    inline void geqp3 ( const int_t            n,                       \
                        const int_t            m,                       \
                        type *                 A,                       \
                        const int_t            ldA,                     \
                        int_t *                jpvt,                    \
                        type *                 tau,                     \
                        type *                 work,                    \
                        int_t                  lwork,                   \
                        real_type_t< type > *  rwork,                   \
                        int_t &                info ) {                 \
        info = 0;                                                       \
        func( & n, & m, A, & ldA, jpvt, tau, work, & lwork, rwork, & info ); }

HLRCOMPRESS_GEQP3_FUNC( std::complex< float >,  cgeqp3_ )
HLRCOMPRESS_GEQP3_FUNC( std::complex< double >, zgeqp3_ )

#undef HLRCOMPRESS_GEQP3_FUNC

//
// *syev/*heev
//
#define HLRCOMPRESS_HEEV_FUNC( type, func )                        \
    inline void heev ( const char   jobz,                          \
                       const char   uplo,                          \
                       const int_t  n,                             \
                       type *       A,                             \
                       const int_t  ldA,                           \
                       type *       W,                             \
                       type *       work,                          \
                       const int_t  lwork,                         \
                       type *,                                     \
                       int_t &      info ) {                       \
        info = 0;                                                  \
        func( & jobz, & uplo, & n, A, & ldA, W, work, & lwork, & info ); }

HLRCOMPRESS_HEEV_FUNC( float,  ssyev_ )
HLRCOMPRESS_HEEV_FUNC( double, dsyev_ )

#undef HLRCOMPRESS_HEEV_FUNC

//
// *syevx/*heevx
//
#define HLRCOMPRESS_HEEVX_FUNC( type, func )                      \
    inline void heevx ( const char             jobz,              \
                        const char             uplo,              \
                        const int_t            n,                 \
                        type *                 A,                 \
                        const int_t            ldA,               \
                        const int_t            il,                \
                        const int_t            iu,                \
                        int_t &                m,                 \
                        real_type_t< type > *  W,                 \
                        type *                 Z,                 \
                        const int_t            ldZ,               \
                        type *                 work,              \
                        const int_t            lwork,             \
                        int_t *                iwork,             \
                        int_t &                info ) {           \
        char                       range = 'I';                   \
        real_type_t< type >  vl, vu;                        \
        real_type_t< type >  abstol = 0;                    \
        int_t                      ifail  = 0;                    \
        info = 0;                                                 \
        func( & jobz, & range, & uplo, & n, A, & ldA, & vl, & vu, & il, & iu, \
              & abstol, & m, W, Z, & ldZ, work, & lwork, iwork, & ifail, & info ); }

HLRCOMPRESS_HEEVX_FUNC( float,  ssyevx_ )
HLRCOMPRESS_HEEVX_FUNC( double, dsyevx_ )

#undef HLRCOMPRESS_HEEVX_FUNC

//
// sstev/dstev
//
#define HLRCOMPRESS_STEV_FUNC( type, func )                \
    inline void stev ( const char   jobz,                  \
                       const int_t  n,                     \
                       type *       D,                     \
                       type *       E,                     \
                       type *       Z,                     \
                       const int_t  ldZ,                   \
                       type *       work,                  \
                       int_t &      info ) {               \
        info = 0;                                          \
        func( & jobz, & n, D, E, Z, & ldZ, work, & info ); }

HLRCOMPRESS_STEV_FUNC( float,  sstev_ )
HLRCOMPRESS_STEV_FUNC( double, dstev_ )

#undef HLRCOMPRESS_STEV_FUNC

//
// *gesvd
//
#define HLRCOMPRESS_GESVD_FUNC( type, func, flops )               \
    inline void gesvd ( const char             jobu,              \
                        const char             jobv,              \
                        const int_t            n,                 \
                        const int_t            m,                 \
                        type *                 A,                 \
                        const int_t            ldA,               \
                        real_type_t< type > *  S,                 \
                        type *                 U,                 \
                        const int_t            ldU,               \
                        type *                 VT,                \
                        const int_t            ldVT,              \
                        type *                 work,              \
                        const int_t            lwork,             \
                        real_type_t< type > *,                    \
                        int_t &                info ) {           \
        info = 0;                                                 \
        func( & jobu, & jobv, & n, & m, A, & ldA, S, U, & ldU, VT, & ldVT, work, & lwork, & info ); }

HLRCOMPRESS_GESVD_FUNC( float,  sgesvd_, SGESVD )
HLRCOMPRESS_GESVD_FUNC( double, dgesvd_, DGESVD )

#undef  HLRCOMPRESS_GESVD_FUNC

#define HLRCOMPRESS_GESVD_FUNC( type, func, flops )               \
    inline void gesvd ( const char             jobu,              \
                        const char             jobv,              \
                        const int_t            n,                 \
                        const int_t            m,                 \
                        type *                 A,                 \
                        const int_t            ldA,               \
                        real_type_t< type > *  S,                 \
                        type *                 U,                 \
                        const int_t            ldU,               \
                        type *                 VT,                \
                        const int_t            ldVT,              \
                        type *                 work,              \
                        const int_t            lwork,             \
                        real_type_t< type > *  rwork,             \
                        int_t &                info ) {           \
        info = 0;                                                 \
        func( & jobu, & jobv, & n, & m, A, & ldA, S, U, & ldU, VT, & ldVT, work, & lwork, rwork, & info ); }

HLRCOMPRESS_GESVD_FUNC( std::complex< float >,  cgesvd_, CGESVD )
HLRCOMPRESS_GESVD_FUNC( std::complex< double >, zgesvd_, CGESVD )

#undef  HLRCOMPRESS_GESVD_FUNC

//
// *gesdd
//
#define HLRCOMPRESS_GESDD_FUNC( type, func, flops )               \
    inline void gesdd ( const char             jobz,              \
                        const int_t            n,                 \
                        const int_t            m,                 \
                        type *                 A,                 \
                        const int_t            ldA,               \
                        real_type_t< type > *  S,                 \
                        type *                 U,                 \
                        const int_t            ldU,               \
                        type *                 VT,                \
                        const int_t            ldVT,              \
                        type *                 work,              \
                        const int_t            lwork,             \
                        real_type_t< type > *,                    \
                        int_t *                iwork,             \
                        int_t &                info ) {           \
        info = 0;                                                 \
        func( & jobz, & n, & m, A, & ldA, S, U, & ldU, VT, & ldVT, work, & lwork, iwork, & info ); }

HLRCOMPRESS_GESDD_FUNC( float,  sgesdd_, SGESDD )
HLRCOMPRESS_GESDD_FUNC( double, dgesdd_, DGESDD )

#undef  HLRCOMPRESS_GESDD_FUNC

#define HLRCOMPRESS_GESDD_FUNC( type, func, flops )               \
    inline void gesdd ( const char             jobz,              \
                        const int_t            n,                 \
                        const int_t            m,                 \
                        type *                 A,                 \
                        const int_t            ldA,               \
                        real_type_t< type > *  S,                 \
                        type *                 U,                 \
                        const int_t            ldU,               \
                        type *                 VT,                \
                        const int_t            ldVT,              \
                        type *                 work,              \
                        const int_t            lwork,             \
                        real_type_t< type > *  rwork,             \
                        int_t *                iwork,             \
                        int_t &                info ) {           \
        info = 0;                                                 \
        func( & jobz, & n, & m, A, & ldA, S, U, & ldU, VT, & ldVT, work, & lwork, rwork, iwork, & info ); }

HLRCOMPRESS_GESDD_FUNC( std::complex< float >,  cgesdd_, CGESDD )
HLRCOMPRESS_GESDD_FUNC( std::complex< double >, zgesdd_, ZGESDD )

#undef  HLRCOMPRESS_GESDD_FUNC

#define HLRCOMPRESS_GESVJ_FUNC( type, func )                       \
    inline void gesvj ( const char             joba,               \
                        const char             jobu,               \
                        const char             jobv,               \
                        const int_t            n,                  \
                        const int_t            m,                  \
                        type *                 A,                  \
                        const int_t            ldA,                \
                        real_type_t< type > *  S,                  \
                        const int_t            mv,                 \
                        type *                 V,                  \
                        const int_t            ldV,                \
                        type *                 cwork,              \
                        const int_t            lwork,              \
                        real_type_t< type > *,                     \
                        const int_t,                               \
                        int_t &                info ) {            \
        info = 0;                                                  \
        func( & joba, & jobu, & jobv, & n, & m, A, & ldA, S, & mv, V, & ldV, cwork, & lwork, & info ); }

HLRCOMPRESS_GESVJ_FUNC( float,  sgesvj_ )
HLRCOMPRESS_GESVJ_FUNC( double, dgesvj_ )

#undef  HLRCOMPRESS_GESVJ_FUNC

#define HLRCOMPRESS_GESVJ_FUNC( type, func )                       \
    inline void gesvj ( const char             joba,               \
                        const char             jobu,               \
                        const char             jobv,               \
                        const int_t            n,                  \
                        const int_t            m,                  \
                        type *                 A,                  \
                        const int_t            ldA,                \
                        real_type_t< type > *  S,                  \
                        const int_t            mv,                 \
                        type *                 V,                  \
                        const int_t            ldV,                \
                        type *                 cwork,              \
                        const int_t            lwork,              \
                        real_type_t< type > *  rwork,              \
                        const int_t            lrwork,             \
                        int_t &                info ) {            \
        info = 0;                                                  \
        func( & joba, & jobu, & jobv, & n, & m, A, & ldA, S, & mv, V, & ldV, cwork, & lwork, rwork, & lrwork, & info ); }

HLRCOMPRESS_GESVJ_FUNC( std::complex< float >,  cgesvj_ )
HLRCOMPRESS_GESVJ_FUNC( std::complex< double >, zgesvj_ )

#undef  HLRCOMPRESS_GESVJ_FUNC

//
// *larfg
//
#define HLRCOMPRESS_LARFG_FUNC( type, func )   \
    inline void larfg ( const int_t  n,        \
                        type &       alpha,    \
                        type *       x,        \
                        const int_t  incx,     \
                        type &       tau ) {   \
        func( & n, & alpha, x, & incx, & tau ); }

HLRCOMPRESS_LARFG_FUNC( float,                  slarfg_ )
HLRCOMPRESS_LARFG_FUNC( double,                 dlarfg_ )
HLRCOMPRESS_LARFG_FUNC( std::complex< float >,  clarfg_ )
HLRCOMPRESS_LARFG_FUNC( std::complex< double >, zlarfg_ )

#undef HLRCOMPRESS_LARFG_FUNC

//
// *larf
//
// apply householder reflection
#define HLRCOMPRESS_LARF_FUNC( type, func )                        \
    inline void larf ( const char   side,                          \
                       const int_t  n,                             \
                       const int_t  m,                             \
                       const type * V,                             \
                       const int_t  incv,                          \
                       const type   tau,                           \
                       type *       C,                             \
                       const int_t  ldc,                           \
                       type *       work ) {                       \
        func( & side, & n, & m, V, & incv, & tau, C, & ldc, work ); }

HLRCOMPRESS_LARF_FUNC( float,                  slarf_ )
HLRCOMPRESS_LARF_FUNC( double,                 dlarf_ )
HLRCOMPRESS_LARF_FUNC( std::complex< float >,  clarf_ )
HLRCOMPRESS_LARF_FUNC( std::complex< double >, zlarf_ )

#undef HLRCOMPRESS_LARF_FUNC

}}// namespace hlrcompress::blas

#endif // __HLRCOMPRESS_BLAS_BLAS_DEF_HH
