#ifndef __HLRCOMPRESS_BLAS_ARITH_HH
#define __HLRCOMPRESS_BLAS_ARITH_HH
//
// Project     : HLRcompress
// Module      : blas/arith
// Description : dense arithmetic based on BLAS/LAPACK
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <vector>
#include <algorithm>

#include <hlrcompress/blas/blas_def.hh>
#include <hlrcompress/blas/vector.hh>
#include <hlrcompress/blas/matrix.hh>

namespace hlrcompress { namespace blas {

////////////////////////////////////////////////////////////////
//
// vector Algebra
//
////////////////////////////////////////////////////////////////

//
// fill vector with constant
//
template < typename T1,
           typename T2 >
std::enable_if_t< is_vector_v< T2 > && std::is_same_v< T1, typename T2::value_t >, void >
fill ( const T1  f,
       T2 &      x )
{
    const idx_t  n = idx_t(x.length());

    for ( idx_t  i = 0; i < n; ++i )
        x(i) = f;
}

//
// conjugate entries in vector
//
template < typename T1 >
std::enable_if_t< is_vector_v< T1 >, void >
conj ( T1 &  x )
{
    if constexpr ( is_complex_type_v< typename T1::value_t > )
    {
        const idx_t  n = idx_t(x.length());
        
        for ( idx_t i = 0; i < n; ++i )
            x(i) = std::conj( x(i) );
    }// if
}

//
// scale vector by constant
//
template < typename T1,
           typename T2 >
std::enable_if_t< is_vector_v< T2 > &&
                  std::is_same_v< T1, typename T2::value_t >,
                  void >
scale ( const T1  f,
        T2 &      x )
{
    scal( int_t(x.length()),
          f,
          x.data(),
          int_t(x.stride()) );
}

//
// copy x into y
//
template < typename T1,
           typename T2 >
std::enable_if_t< is_vector_v< T1 > &&
                  is_vector_v< T2 > &&
                  std::is_same_v< typename T1::value_t, typename T2::value_t >,
                  void >
copy ( const T1 &  x,
       T2 &        y )
{
    HLRCOMPRESS_DBG_ASSERT( x.length() == y.length() );

    copy( int_t(x.length()),
          x.data(),
          int_t(x.stride()),
          y.data(),
          int_t(y.stride()) );
}

//
// exchange x and y
//
template < typename T1,
           typename T2 >
std::enable_if_t< is_vector_v< T1 > &&
                  is_vector_v< T2 > &&
                  std::is_same_v< typename T1::value_t, typename T2::value_t >,
                  void >
swap ( T1 &  x,
       T2 &  y )
{
    HLRCOMPRESS_DBG_ASSERT( x.length() == y.length() );

    swap( int_t(x.length()),
          x.data(),
          int_t(x.stride()),
          y.data(),
          int_t(y.stride()) );
}

//
// determine index with maximal absolute value in x
//
template < typename T1 >
std::enable_if_t< is_vector_v< T1 >, idx_t >
max_idx ( const T1 &  x )
{
    HLRCOMPRESS_DBG_ASSERT( x.length() > 0 );

    auto  res = idx_t( max_idx( int_t(x.length()),
                                x.data(),
                                int_t(x.stride()) ) - 1 );

    return res;
}

//
// determine index with minimax absolute value in x
//
template < typename T1 >
std::enable_if_t< is_vector_v< T1 >, idx_t >
min_idx ( const T1 &  x )
{
    HLRCOMPRESS_DBG_ASSERT( x.length() > 0 );

    using  value_t = typename T1::value_t;
    using  real_t  = real_type_t< value_t >;
    
    const idx_t  n       = idx_t(x.length());
    real_t       min_val = std::abs( x(0) );
    idx_t        min_idx = 0;

    for ( idx_t  i = 1; i < n; ++i )
    {
        const real_t  val = std::abs( x(i) );
        
        if ( val < min_val )
        {
            min_val = val;
            min_idx = i;
        }// if
    }// for

    return min_idx;
}

//
// compute y ≔ y + α·x
//
template < typename T1,
           typename T2,
           typename T3>
std::enable_if_t< is_vector_v< T2 > &&
                  is_vector_v< T3 > &&
                  std::is_same_v< T1, typename T2::value_t > &&
                  std::is_same_v< T1, typename T3::value_t >,
                  void >
add ( const T1    alpha,
      const T2 &  x,
      T3 &        y )
{
    HLRCOMPRESS_DBG_ASSERT( x.length() == y.length() );
    
    axpy( int_t(x.length()),
          alpha,
          x.data(),
          int_t(x.stride()),
          y.data(),
          int_t(y.stride()) );
}

//
// compute <x,y> = x^H · y
//
template < typename T1,
           typename T2 >
std::enable_if_t< is_vector_v< T1 > &&
                  is_vector_v< T2 > &&
                  std::is_same_v< typename T1::value_t, typename T2::value_t >,
                  typename T1::value_t >
dot ( const T1 &  x,
      const T2 &  y )
{
    HLRCOMPRESS_DBG_ASSERT( x.length() == y.length() );

    auto  res = dot( int_t(x.length()),
                     x.data(),
                     int_t(x.stride()),
                     y.data(),
                     int_t(y.stride()) );

    return res;
}

//
// compute <x,y> without conjugating x, e.g. x^T · y
//
template < typename T1,
           typename T2 >
std::enable_if_t< is_vector_v< T1 > &&
                  is_vector_v< T2 > &&
                  std::is_same_v< typename T1::value_t, typename T2::value_t >,
                  typename T1::value_t >
dotu ( const T1 &  x,
       const T2 &  y )
{
    HLRCOMPRESS_DBG_ASSERT( x.length() == y.length() );

    auto  res = dotu( int_t(x.length()),
                      x.data(),
                      int_t(x.stride()),
                      y.data(),
                      int_t(y.stride()) );

    return res;
}

// helper for stable_dotu
template < typename T1 >
bool
abs_lt ( const T1  a1,
         const T1  a2 )
{
    return std::abs( a1 ) < std::abs( a2 );
}

//
// compute dot product x · y numerically stable
// \param  x  first argument of dot product
// \param  y  second argument of dot product
//
template < typename T1,
           typename T2 >
std::enable_if_t< is_vector_v< T1 > &&
                  is_vector_v< T2 > &&
                  std::is_same_v< typename T1::value_t, typename T2::value_t >,
                  typename T1::value_t >
stable_dotu ( const T1 &  x,
              const T2 &  y )
{
    HLRCOMPRESS_DBG_ASSERT( x.length() == y.length() );

    using  value_t = typename T1::value_t;

    std::vector< value_t >  tmp( x.length() );
    const idx_t             n = idx_t(x.length());

    // compute conj(x_i) · y_i for each i
    for ( idx_t  i = 0; i < n; ++i )
        tmp[i] = x(i) * y(i);

    // sort tmp w.r.t. absolute value of element
    std::sort( tmp.begin(), tmp.end(), abs_lt< value_t > );

    // compute final result
    value_t  res = value_t(0);

    for ( idx_t  i = 0; i < n; ++i )
        res += tmp[i];

    return res;
}

//
// compute sum of elements in x numerically stable
// \param  x  vector holding coefficients to sum up
//
template < typename T1 >
std::enable_if_t< is_vector_v< T1 >,
                  typename T1::value_t >
stable_sum ( const T1 &  x )
{
    using  value_t = typename T1::value_t;
    
    std::vector< value_t >  tmp( x.length() );
    const idx_t             n = idx_t(x.length());

    // compute conj(x_i) · y_i for each i
    for ( idx_t  i = 0; i < n; ++i )
        tmp[i] = x(i);

    // sort tmp w.r.t. absolute value of element
    std::sort( tmp.begin(), tmp.end(), abs_lt< value_t > );

    // compute final result
    value_t  res = value_t(0);

    for ( idx_t  i = 0; i < n; ++i )
        res += tmp[i];

    return res;
}

//
// compute ∥x∥₂
//
template < typename T1 >
typename std::enable_if< is_vector_v< T1 >,
                         real_type_t< typename T1::value_t > >::type
norm2 ( const T1 &  x )
{
    auto  res = norm2( int_t(x.length()),
                       x.data(),
                       int_t(x.stride()) );

    return res;
}

template < typename T1 >
std::enable_if_t< is_vector_v< T1 >,
                  real_type_t< typename T1::value_t > >
norm_2 ( const T1 &  x )
{
    return norm2( x );
}

////////////////////////////////////////////////////////////////
//
// Basic matrix Algebra
//
////////////////////////////////////////////////////////////////

//
// set M to f entrywise
//
template < typename T1,
           typename T2 >
std::enable_if_t< is_matrix_v< T2 > &&
                  std::is_same_v< T1, typename T2::value_t >,
                  void >
fill ( const T1  f,
       T2 &      M )
{
    const idx_t  n = idx_t(M.nrows());
    const idx_t  m = idx_t(M.ncols());
    
    for ( idx_t j = 0; j < m; ++j )
        for ( idx_t i = 0; i < n; ++i )
            M(i,j) = f;
}

//
// conjugate entries in vector
//
template < typename T1 >
std::enable_if_t< is_matrix_v< T1 >, void >
conj ( T1 &  M )
{
    using  value_t = typename T1::value_t;
    
    if constexpr ( is_complex_type_v< value_t > )
    {
        const idx_t  n = idx_t(M.nrows());
        const idx_t  m = idx_t(M.ncols());
    
        for ( idx_t j = 0; j < m; ++j )
            for ( idx_t i = 0; i < n; ++i )
                M( i, j ) = std::conj( M( i, j ) );
    }// if
}

//
// compute M ≔ f · M
//
template < typename T1,
           typename T2 >
std::enable_if_t< is_matrix_v< T2 > &&
                  std::is_same_v< T1, typename T2::value_t >,
                  void >
scale ( const T1  f,
        T2 &      M )
{
    const idx_t  n = idx_t(M.nrows());
    const idx_t  m = idx_t(M.ncols());
    
    for ( idx_t j = 0; j < m; ++j )
        for ( idx_t i = 0; i < n; ++i )
            M(i,j) *= f;
}

//
// copy A to B
//
template < typename T1,
           typename T2 >
std::enable_if_t< is_matrix_v< T1 > &&
                  is_matrix_v< T2 >,
                  void >
copy ( const T1 &  A,
       T2 &        B )
{
    HLRCOMPRESS_DBG_ASSERT( A.nrows() == B.nrows() );
    HLRCOMPRESS_DBG_ASSERT( A.ncols() == B.ncols() );

    using  value_dest_t = value_type_t< T2 >;
    
    const idx_t  n = idx_t(A.nrows());
    const idx_t  m = idx_t(A.ncols());
    
    for ( idx_t j = 0; j < m; ++j )
        for ( idx_t i = 0; i < n; ++i )
            B(i,j) = value_dest_t( A(i,j) );
}

template < typename matrix_t >
typename std::enable_if_t< is_matrix< matrix_t >::value,
                           matrix< typename matrix_t::value_t > >
copy ( const matrix_t &  A )
{
    using  value_t = typename matrix_t::value_t;

    auto  M = matrix< value_t >( A.nrows(), A.ncols() );

    copy( A, M );

    return M;
}

template < typename value_dest_t,
           typename value_src_t >
matrix< value_dest_t >
copy ( const matrix< value_src_t > &  A )
{
    matrix< value_dest_t >  M( A.nrows(), A.ncols() );
    const size_t            n = M.nrows() * M.ncols();

    for ( size_t  i = 0; i < n; ++i )
        M.data()[i] = value_dest_t( A.data()[i] );

    return M;
}

//
// transpose matrix A: A → A^T
//        ASSUMPTION: A is square matrix
//
template < typename T1 >
std::enable_if_t< is_matrix_v< T1 >, void >
transpose ( T1 &  A )
{
    HLRCOMPRESS_DBG_ASSERT( A.nrows() == A.ncols() );

    const idx_t  n = idx_t(A.nrows());

    for ( idx_t  j = 0; j < n-1; ++j )
        for ( idx_t i = j+1; i < n; ++i )
        {
            const auto  t = A(j,i);

            A(j,i) = A(i,j);
            A(i,j) = t;
        }// for
}
            
//
// conjugate transpose matrix A: A → A^H
//        ASSUMPTION: A is square matrix
//
template < typename T1 >
std::enable_if_t< is_matrix_v< T1 >, void >
conj_transpose ( T1 &  A )
{
    HLRCOMPRESS_DBG_ASSERT( A.nrows() == A.ncols() );

    const idx_t  n = idx_t(A.nrows());

    if constexpr ( is_complex_type_v< typename T1::value_t > )
    {
        for ( idx_t  j = 0; j < n-1; ++j )
            for ( idx_t i = j+1; i < n; ++i )
            {
                const auto  t = A(j,i);
                
                A(j,i) = std::conj( A(i,j) );
                A(i,j) = std::conj( t );
            }// for
    }// if
    else
    {
        for ( idx_t  j = 0; j < n-1; ++j )
            for ( idx_t i = j+1; i < n; ++i )
                std::swap( A(i,j), A(j,i) );
    }// if
}
            
//
// determine index (i,j) with maximal absolute value in M
//        and return in row and col
//
template < typename T1 >
std::enable_if_t< is_matrix_v< T1 >, void >
max_idx ( const T1 &  M,
          idx_t &     row,
          idx_t &     col )
{
    using  value_t = typename T1::value_t;
    using  real_t  = real_type_t< value_t >;
    
    HLRCOMPRESS_DBG_ASSERT( M.nrows() > 0 );
    HLRCOMPRESS_DBG_ASSERT( M.ncols() > 0 );

    real_t  max_val = real_t(0);

    // init with forbidden values
    idx_t   lrow    = idx_t(M.nrows());
    idx_t   lcol    = idx_t(M.ncols());
    
    for ( idx_t  j = 0; j < idx_t(M.ncols()); ++j )
        for ( idx_t  i = 0; i < idx_t(M.nrows()); ++i )
        {
            const real_t  val = std::abs( M(i,j) );

            if ( val > max_val )
            {
                max_val = val;
                lrow    = i;
                lcol    = j;
            }// if
        }// for

    row = lrow;
    col = lcol;
}

//
// compute B = B + f A
//                                                     
template < typename T1,
           typename T2,
           typename T3 >
std::enable_if_t< is_matrix_v< T2 > &&
                  is_matrix_v< T3 > &&
                  std::is_same_v< T1, typename T2::value_t > &&
                  std::is_same_v< T1, typename T3::value_t >,
                  void >
add ( const T1    f,
      const T2 &  A,
      T3 &        B )
{
    HLRCOMPRESS_DBG_ASSERT( A.nrows() == B.nrows() );
    HLRCOMPRESS_DBG_ASSERT( A.ncols() == B.ncols() );

    // using  value_t = T1;
    
    const idx_t  n = idx_t(A.nrows());
    const idx_t  m = idx_t(A.ncols());
    
    for ( idx_t j = 0; j < m; ++j )
        for ( idx_t i = 0; i < n; ++i )
            B(i,j) += f * A(i,j);
}

//!
//! \ingroup  BLAS_Module
//! \brief compute A ≔ A + α·x·y^H
//!
template < typename value_t >
void
add_r1 ( const value_t              alpha,
         const vector< value_t > &  x,
         const vector< value_t > &  y,
         matrix< value_t > &        A )
{
    HLRCOMPRESS_DBG_ASSERT( x.length() == A.nrows() );
    HLRCOMPRESS_DBG_ASSERT( y.length() == A.ncols() );

    if ( A.row_stride() == 1 )
    {
        ger( int_t(A.nrows()),
             int_t(A.ncols()),
             alpha,
             x.data(),
             int_t(x.stride()),
             y.data(),
             int_t(y.stride()),
             A.data(),
             int_t(A.col_stride()) );
    }// if
    else
    {
        const idx_t  n = idx_t(A.nrows());
        const idx_t  m = idx_t(A.ncols());

        if constexpr ( is_complex_type_v< value_t > )
        {
            for ( idx_t j = 0; j < m; ++j )
            {
                const value_t  f = alpha * std::conj( y(j) );
    
                for ( idx_t i = 0; i < n; ++i )
                    A(i,j) += x(i) * f;
            }// for
        }// if
        else
        {
            for ( idx_t j = 0; j < m; ++j )
            {
                const value_t  f = alpha * y(j);
    
                for ( idx_t i = 0; i < n; ++i )
                    A(i,j) += x(i) * f;
            }// for
        }// else
    }// else
}

//
// compute y ≔ β·y + α·A·x
//
template < typename T1,
           typename T2,
           typename T3,
           typename T4 >
std::enable_if_t< is_matrix_v< T2 > &&
                  is_vector_v< T3 > &&
                  is_vector_v< T4 > &&
                  std::is_same_v< T1, typename T2::value_t > &&
                  std::is_same_v< T1, typename T3::value_t > &&
                  std::is_same_v< T1, typename T4::value_t >,
                  void >
mulvec ( const T1    alpha,
         const T2 &  A,
         const T3 &  x,
         const T1    beta,
         T4 &        y )
{
    HLRCOMPRESS_DBG_ASSERT( x.length() == A.ncols() );
    HLRCOMPRESS_DBG_ASSERT( y.length() == A.nrows() );

    using  value_t = T1;

    // if (( A.row_stride() == 1 ) && ( std::max( A.nrows(), A.ncols() ) > 8 ))
    if ( A.row_stride() == 1 )
    {
        gemv( char( A.blas_view() ),
              int_t(A.blas_nrows()),
              int_t(A.blas_ncols()),
              alpha,
              A.data(),
              int_t(A.col_stride()),
              x.data(),
              int_t(x.stride()),
              beta,
              y.data(),
              int_t(y.stride()) );
    }// if
    else
    {
        const idx_t  n = idx_t(A.nrows());
        const idx_t  m = idx_t(A.ncols());
        
        if ( beta == value_t(0) )
        {
            for ( idx_t i = 0; i < n; ++i )
                y(i) = value_t(0);
        }// if
        else if ( beta != value_t(1) )
        {
            for ( idx_t i = 0; i < n; ++i )
                y(i) *= beta;
        }// if
        
        for ( idx_t i = 0; i < n; ++i )
        {
            value_t  f = value_t(0);
            
            for ( idx_t j = 0; j < m; ++j )
                f += A(i,j) * x(j);
            
            y(i) += alpha * f;
        }// for
    }// else
}

//
// compute y ≔ α·A·x
//
template < typename T1,
           typename T2,
           typename T3 >
std::enable_if_t< is_matrix_v< T2 > &&
                  is_vector_v< T3 > &&
                  std::is_same_v< T1, typename T2::value_t > &&
                  std::is_same_v< T1, typename T3::value_t >,
                  vector< typename T2::value_t > >
mulvec ( const T1    alpha,
         const T2 &  A,
         const T3 &  x )
{
    HLRCOMPRESS_DBG_ASSERT( x.length() == A.ncols() );

    using  value_t = T1;

    vector< value_t >  y( A.nrows() );
    
    // if (( A.row_stride() == 1 ) && ( std::max( A.nrows(), A.ncols() ) > 8 ))
    if ( A.row_stride() == 1 )
    {
        gemv( char( A.blas_view() ),
              int_t(A.blas_nrows()),
              int_t(A.blas_ncols()),
              alpha,
              A.data(),
              int_t(A.col_stride()),
              x.data(),
              int_t(x.stride()),
              value_t(1),
              y.data(),
              int_t(y.stride()) );
    }// if
    else
    {
        const idx_t  n = idx_t(A.nrows());
        const idx_t  m = idx_t(A.ncols());
        
        for ( idx_t i = 0; i < n; ++i )
        {
            value_t  f = value_t(0);
            
            for ( idx_t j = 0; j < m; ++j )
                f += A(i,j) * x(j);
            
            y(i) += alpha * f;
        }// for
    }// else

    return y;
}

//
// compute C ≔ β·C + α·A·B
//
template < typename T1,
           typename T2,
           typename T3,
           typename T4 >
std::enable_if_t< is_matrix_v< T2 > &&
                  is_matrix_v< T3 > &&
                  is_matrix_v< T4 > &&
                  std::is_same_v< T1, typename T2::value_t > &&
                  std::is_same_v< T1, typename T3::value_t > &&
                  std::is_same_v< T1, typename T4::value_t >,
                  void >
prod ( const T1    alpha,
       const T2 &  A,
       const T3 &  B,
       const T1    beta,
       T4 &        C )
{
    HLRCOMPRESS_DBG_ASSERT( A.nrows() == C.nrows() );
    HLRCOMPRESS_DBG_ASSERT( B.ncols() == C.ncols() );
    HLRCOMPRESS_DBG_ASSERT( A.ncols() == B.nrows() );
    HLRCOMPRESS_DBG_ASSERT( A.col_stride() != 0    );
    HLRCOMPRESS_DBG_ASSERT( B.col_stride() != 0    );
    HLRCOMPRESS_DBG_ASSERT( C.col_stride() != 0    );
    
    gemm( char( A.blas_view() ),
          char( B.blas_view() ),
          int_t(C.nrows()),
          int_t(C.ncols()),
          int_t(A.ncols()),
          alpha, A.data(),
          int_t(A.col_stride()),
          B.data(),
          int_t(B.col_stride()),
          beta,
          C.data(),
          int_t(C.col_stride()) );
}

//
// compute C ≔ α·A·B
//
template < typename T1,
           typename T2,
           typename T3 >
std::enable_if_t< is_matrix_v< T2 > &&
                  is_matrix_v< T3 > &&
                  std::is_same_v< T1, typename T2::value_t > &&
                  std::is_same_v< T1, typename T3::value_t >,
                  matrix< typename T2::value_t > >
prod ( const T1    alpha,
       const T2 &  A,
       const T3 &  B )
{
    HLRCOMPRESS_DBG_ASSERT( A.ncols() == B.nrows() );
    HLRCOMPRESS_DBG_ASSERT( A.col_stride() != 0    );
    HLRCOMPRESS_DBG_ASSERT( B.col_stride() != 0    );
    
    using  value_t = T1;

    auto  C = matrix< value_t >( A.nrows(), B.ncols() );

    gemm( char( A.blas_view() ),
          char( B.blas_view() ),
          int_t(C.nrows()),
          int_t(C.ncols()),
          int_t(A.ncols()),
          alpha, A.data(),
          int_t(A.col_stride()),
          B.data(),
          int_t(B.col_stride()),
          value_t(1),  // C is initialised with zero
          C.data(),
          int_t(C.col_stride()) );

    return C;
}

//
// multiply k columns of M with diagonal matrix D,
//           e.g. compute M ≔ M·D
//
template < typename T1,
           typename T2 >
std::enable_if_t< is_matrix_v< T1 > &&
                  is_vector_v< T2 > &&
                  std::is_same_v< real_type_t< typename T1::value_t >,
                                  real_type_t< typename T2::value_t > >,
                  void >
prod_diag ( T1 &         M,
            const T2 &   D,
            const idx_t  k )
{
    using  value_t = typename T1::value_t;
    
    const range  row_is( 0, idx_t( M.nrows() )-1 );
    
    for ( idx_t  i = 0; i < k; ++i )
    {
        vector< value_t >  Mi( M, row_is, i );

        scale( value_t(D(i)), Mi );
    }// for
}

//
// multiply k rows of M with diagonal matrix D,
//           e.g. compute M ≔ D·M
//
template < typename T1,
           typename T2 >
std::enable_if_t< is_vector_v< T1 > &&
                  is_matrix_v< T2 > &&
                  std::is_same_v< real_type_t< typename T1::value_t >,
                                  real_type_t< typename T2::value_t > >,
                  void >
prod_diag ( const T1 &   D,
            T2 &         M,
            const idx_t  k )
{
    using  value_t = typename T1::value_t;
    
    const range  col_is( 0, idx_t( M.ncols() )-1 );
    
    for ( idx_t  i = 0; i < k; ++i )
    {
        vector< value_t >  Mi( M, i, col_is );

        scale( value_t(D(i)), Mi );
    }// for
}

//
// return spectral norm of M
//
template < typename T1 >
std::enable_if_t< is_matrix_v< T1 >,
                  real_type_t< typename T1::value_t > >
norm2 ( const T1 & M );

template < typename T1 >
std::enable_if_t< is_matrix_v< T1 >,
                  real_type_t< typename T1::value_t > >
norm_2 ( const T1 & M )
{
    return norm2( M );
}

//
// return Frobenius norm of M
//
template < typename T1 >
std::enable_if_t< is_matrix_v< T1 >,
                  real_type_t< typename T1::value_t > >
normF ( const T1 & M )
{
    using  value_t = typename T1::value_t;
    
    const idx_t  n = idx_t(M.nrows());
    const idx_t  m = idx_t(M.ncols());

    if ( M.col_stride() == n * M.row_stride() )
    {
        return norm2( int_t(n*m),
                      M.data(),
                      int_t(M.row_stride()) );
    }// if
    else
    {
        value_t  f = value_t(0);

        if constexpr( is_complex_type_v< typename T1::value_t > )
        {
            for ( idx_t j = 0; j < m; ++j )
                for ( idx_t i = 0; i < n; ++i )
                    f += std::conj( M(i,j) ) * M(i,j);
        
            return std::real( std::sqrt( f ) );
        }// if
        else
        {
            for ( idx_t j = 0; j < m; ++j )
                for ( idx_t i = 0; i < n; ++i )
                    f += M(i,j) * M(i,j);
        
            return std::sqrt( f );
        }// if
    }// else
}

template < typename T1 >
std::enable_if_t< is_matrix_v< T1 >,
                  real_type_t< typename T1::value_t > >
norm_F ( const T1 & M )
{
    return normF( M );
}

//
// compute Frobenius norm of A-B
//
template < typename T1,
           typename T2 >
std::enable_if_t< is_matrix_v< T1 > &&
                  is_matrix_v< T2 > &&
                  std::is_same_v< typename T1::value_t, typename T2::value_t >,
                  real_type_t< typename T1::value_t > >
diff_normF ( const T1 &  A,
             const T2 &  B )
{
    HLRCOMPRESS_DBG_ASSERT( A.nrows() == B.nrows() );
    HLRCOMPRESS_DBG_ASSERT( A.ncols() == B.ncols() );
    
    using  value_t = typename T1::value_t;

    const idx_t  n = idx_t(A.nrows());
    const idx_t  m = idx_t(A.ncols());
    value_t      f = value_t(0);
    
    if constexpr( is_complex_type_v< typename T1::value_t > )
    {
        for ( idx_t j = 0; j < m; ++j )
        {
            for ( idx_t i = 0; i < n; ++i )
            {
                const value_t  a_ij = A(i,j) - B(i,j);
                
                f += std::conj( a_ij ) * a_ij;
            }// for
        }// for
        
        return std::real( std::sqrt( f ) );
    }// if
    else
    {
        for ( idx_t j = 0; j < m; ++j )
        {
            for ( idx_t i = 0; i < n; ++i )
            {
                const value_t  a_ij = A(i,j) - B(i,j);
                
                f += a_ij * a_ij;
            }// for
        }// for
        
        return std::sqrt( f );
    }// else
}

template < typename T1,
           typename T2 >
std::enable_if_t< is_matrix_v< T1 > &&
                  is_matrix_v< T2 > &&
                  std::is_same_v< typename T1::value_t, typename T2::value_t >,
                  real_type_t< typename T1::value_t > >
diff_norm_F ( const T1 &  A,
              const T2 &  B )
{
    return diff_normF( A, B );
}

//
// compute Frobenius norm of M=A·B^H
//
template < typename T1,
           typename T2 >
std::enable_if_t< is_matrix_v< T1 > &&
                  is_matrix_v< T2 > &&
                  std::is_same_v< typename T1::value_t, typename T2::value_t >,
                  real_type_t< typename T1::value_t > >
norm_F ( const T1 &  U,
         const T2 &  V )
{
    HLRCOMPRESS_DBG_ASSERT( U.ncols() != V.ncols() );
    
    using  value_t = typename T1::value_t;

    //
    // ∑_ij (M_ij)² = ∑_ij (∑_k u_ik v_jk')²
    //              = ∑_ij (∑_k u_ik v_jk') (∑_l u_il v_jl')'
    //              = ∑_ij ∑_k ∑_l u_ik v_jk' u_il' v_jl
    //              = ∑_k ∑_l ∑_i u_ik u_il' ∑_j v_jk' v_jl
    //              = ∑_k ∑_l (u_l)^H · u_k  v_k^H · v_l
    //
    
    auto  res = value_t(0);
    
    for ( size_t  k = 0; k < U.ncols(); k++ )
    {
        auto  u_k = U.column( k );
        auto  v_k = V.column( k );
                
        for ( size_t  l = 0; l < V.ncols(); l++ )
        {
            auto  u_l = U.column( l );
            auto  v_l = V.column( l );

            res += dot( u_k, u_l ) * dot( v_k, v_l );
        }// for
    }// for

    return std::abs( std::sqrt( res ) );
}

////////////////////////////////////////////////////////////////
//
// Advanced matrix Algebra
//
////////////////////////////////////////////////////////////////

//
// Compute QR factorisation of the n×m matrix A with
// n×m matrix Q and mxm matrix R (n >= m); A will be
// overwritten with Q upon exit
//
template < typename value_t >
void
qr  ( matrix< value_t > &  M,
      matrix< value_t > &  R,
      const bool           comp_Q = true )
{
    const auto              nrows = M.nrows();
    const auto              ncols = M.ncols();
    const auto              minrc = std::min( nrows, ncols );
    std::vector< value_t >  tau( ncols );
    std::vector< value_t >  work( ncols );
    int_t              info = 0;

    geqr2( nrows, ncols, M.data(), nrows, tau.data(), work.data(), info );

    if (( R.nrows() != minrc ) || ( R.ncols() != ncols ))
        R = std::move( blas::matrix< value_t >( minrc, ncols ) );
    
    if ( comp_Q )
    {
        if ( ncols > nrows )
        {
            //
            // copy M to R, resize M, copy M back and nullify R in
            //

            copy( M, R );
            M = std::move( matrix< value_t >( nrows, nrows ) );

            auto  RM = blas::matrix< value_t >( R, range::all, range( 0, nrows-1 ) );

            copy( RM, M );

            for ( size_t  j = 0; j < nrows; ++j )
                for ( size_t  i = j+1; i < nrows; ++i )
                    R(i,j) = value_t(0);

            ung2r( nrows, nrows, nrows, M.data(), nrows, tau.data(), work.data(), info );
        }// if
        else
        {
            // just copy R from M
            for ( size_t  j = 0; j < ncols; ++j )
                for ( size_t  i = 0; i <= j; ++i )
                    R(i,j) = M(i,j);

            ung2r( nrows, ncols, ncols, M.data(), nrows, tau.data(), work.data(), info );
        }// else
    }// if
    else
    {
        for ( size_t  j = 0; j < ncols; ++j )
            for ( size_t  i = 0; i <= std::min( j, minrc-1 ); ++i )
                R(i,j) = M(i,j);
    }// else
}

//
// Compute QR factorisation of the n×m matrix A, m ≪ n, with
// n×m matrix Q and mxm matrix R (n >= m); A will be
// overwritten with Q upon exit
// - ntile defines tile size (0: use internal default tile size)
//
template < typename T1 >
std::enable_if_t< is_matrix_v< T1 >, void >
tsqr   ( T1 &                              A,
         matrix< typename T1::value_t > &  R,
         const size_t                      ntile = 0 );

//
// Compute QR factorisation with column pivoting of the n×m matrix A,
// i.e., \f$ AP = QR \f$, with n×m matrix Q and mxm matrix R (n >= m).
// P_i = j means, column i of AP was column j of A; A will be
// overwritten with Q upon exit
//
template < typename T1 >
std::enable_if_t< is_matrix_v< T1 >, void >
qrp    ( T1 &                              A,
         matrix< typename T1::value_t > &  R,
         std::vector< int_t > &       P )
{
    using  value_t = value_type_t< T1 >;
    using  real_t  = real_type_t< value_t >;
    
    const auto  n     = int_t( A.nrows() );
    const auto  m     = int_t( A.ncols() );
    const auto  minnm = std::min( n, m );
    int_t  info  = 0;

    if ( int_t( P.size() ) != m )
        P.resize( m );
    
    //
    // workspace query
    //

    value_t  dummy      = value_t(0); // non-NULL workspace for latest Intel MKL
    value_t  work_query = value_t(0);
    auto     rwork      = vector< real_t >( 2*m );

    geqp3( n, m, A.data(), int_t( A.col_stride() ), P.data(),
           & dummy, & work_query, LAPACK_WS_QUERY, rwork.data(), info );

    HLRCOMPRESS_ASSERT( info == 0 );
    
    auto   lwork = int_t( std::real( work_query ) );

    orgqr( n, minnm, minnm, A.data(), int_t( A.col_stride() ), & dummy,
           & work_query, LAPACK_WS_QUERY, info );
    
    HLRCOMPRESS_ASSERT( info == 0 );

    // adjust work space size
    lwork = std::max( lwork, int_t( std::real( work_query ) ) );
    
    auto       tmp_space = vector< value_t >( lwork + m );
    value_t *  work      = tmp_space.data();
    value_t *  tau       = work + lwork;

    //
    // compute Householder vectors and R
    //
    
    geqp3( n, m, A.data(), int_t( A.col_stride() ), P.data(), tau, work, lwork, rwork.data(), info );
    
    HLRCOMPRESS_ASSERT( info == 0 );

    //
    // copy upper triangular matrix to R
    //

    if (( int_t( R.nrows() ) != m ) || ( int_t( R.ncols() ) != m ))
        R = std::move( matrix< value_t >( m, m ) );
    else
        fill( value_t(0), R );
    
    for ( int_t  i = 0; i < m; i++ )
    {
        const auto         irange = range( 0, std::min( i, minnm-1 ) );
        vector< value_t >  colA( A, irange, i );
        vector< value_t >  colR( R, irange, i );

        copy( colA, colR );
    }// for
        
    //
    // compute Q
    //
    
    orgqr( n, minnm, minnm, A.data(), int_t( A.col_stride() ), tau, work, lwork, info );

    if ( n < m )
    {
        //
        // realloc Q
        //

        auto  Q  = matrix< value_t >( n, n );
        auto  Ai = matrix< value_t >( A, range::all, range( 0, n-1 ) );

        copy( Ai, Q );

        A = std::move( Q );
    }// if
    
    HLRCOMPRESS_ASSERT( info == 0 );

    //
    // adjount indices in P (1-counted to 0-counted)
    //

    for ( int_t  i = 0; i < m; i++ )
        --P[i];
}

//
// compute SVD decomposition \f$ A = U·S·V^H \f$ of the nxm matrix A with
// n×min(n,m) matrix U, min(n,m)×min(n,m) matrix S (diagonal)
// and m×min(n,m) matrix V; A will be overwritten with U upon exit
//
template < typename T1 >
std::enable_if_t< is_matrix_v< T1 >, void >
svd    ( T1 &                                             A,
         vector< real_type_t< typename T1::value_t > > &  S,
         matrix< typename T1::value_t > &                 V )
{
    using  value_t = typename T1::value_t;
    using  real_t  = real_type_t< value_t >;

    const int_t    n      = int_t( A.nrows() );
    const int_t    m      = int_t( A.ncols() );
    const int_t    min_nm = std::min( n, m );
    int_t          info   = 0;
    value_t             work_query = value_t(0);
    matrix< value_t >   VT( min_nm, A.ncols() );
    value_t             vdummy = 0;
    real_t              rdummy = 0;

    if ( S.length() != size_t( min_nm ) )
        S = std::move( vector< real_t >( min_nm ) );
    
    // work space query
    gesvd( 'O', 'S',
           int_t( A.nrows() ),
           int_t( A.ncols() ),
           A.data(),
           int_t( A.col_stride() ),
           S.data(),
           & vdummy,
           int_t( A.col_stride() ),
           & vdummy,
           int_t( VT.col_stride() ),
           & work_query,
           LAPACK_WS_QUERY,
           & rdummy,
           info );

    HLRCOMPRESS_ASSERT( info == 0 );
        
    const int_t   lwork = int_t( std::real( work_query ) );
    vector< value_t >  work( lwork );
    vector< real_t >   rwork( 5 * min_nm );
    
    gesvd( 'O', 'S',
           int_t( A.nrows() ),
           int_t( A.ncols() ),
           A.data(),
           int_t( A.col_stride() ),
           S.data(),
           & vdummy,
           int_t( A.col_stride() ),
           VT.data(),
           int_t( VT.col_stride() ),
           work.data(),
           lwork,
           rwork.data(),
           info );

    HLRCOMPRESS_ASSERT( info == 0 );

    if (( V.nrows() != VT.ncols() ) || ( V.ncols() != size_t(min_nm) ))
        V = std::move( matrix< value_t >( VT.ncols(), min_nm ) );
    
    copy( adjoint( VT ), V );
}

//
// compute SVD decomposition \f$ A = U·S·V^H \f$ of the nxm matrix A
// but return only the left/right singular vectors and the
// singular values S ∈ ℝ^min(n,m);
// upon exit, A will be contain the corresponding sing. vectors
//
template < typename T1 >
std::enable_if_t< is_matrix_v< T1 >, void >
svd    ( T1 &                                                      A,
         vector< real_type_t< typename T1::value_t > > &  S,
         const bool                                                left = true );

}}// namespace hlrcompress::blas

#endif // __HLRCOMPRESS_BLAS_ARITH_HH
