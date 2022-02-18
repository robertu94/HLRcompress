#ifndef __HLRCOMPRESS_APPROX_RRQR_HH
#define __HLRCOMPRESS_APPROX_RRQR_HH
//
// Project     : HLRcompress
// Module      : approx/rrqr
// Description : low-rank approximation functions using rank revealing QR
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <utility>

#include <hlrcompress/blas/arith.hh>
#include <hlrcompress/misc/type_traits.hh>
#include <hlrcompress/misc/error.hh>

namespace hlrcompress
{

namespace detail
{

//
// determine truncate rank of R by looking at
// norms of R(i:·,i:·) for all i
//
template < typename value_t >
int
trunc_rank ( const blas::matrix< value_t > &  R,
             const accuracy &                 acc )
{
    using  real_t = real_type_t< value_t >;

    HLRCOMPRESS_ASSERT( R.nrows() == R.ncols() );
    
    const idx_t             n = idx_t( R.nrows() );
    blas::vector< real_t >  s( n );
    
    for ( int  i = 0; i < n; ++i )
    {
        auto  rest = blas::range( i, n-1 );
        auto  R_i  = blas::matrix< value_t >( R, rest, rest );
        
        s( i ) = blas::normF( R_i );
    }// for

    return acc.trunc_rank( s );
}

}// namespace detail

//
// return low-rank approximation of M with accuracy <acc>
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
rrqr ( blas::matrix< value_t > &  M,
       const accuracy &           acc )
{
    //
    // algorithm only works for nrows >= ncols, so proceed with
    // transposed matrix if ncols > nrows
    //

    const idx_t  nrows = idx_t( M.nrows() );
    const idx_t  ncols = idx_t( M.ncols() );

    if ( ncols > nrows )
    {
        //
        // compute RRQR for M^H, e.g., M^H = U·V^H
        // and return V·U^H
        //
        
        auto  MH = blas::matrix< value_t >( ncols, nrows );

        blas::copy( blas::adjoint( M ), MH );

        auto [ U, V ] = rrqr( MH, acc );

        return { std::move( V ), std::move( U ) };
    }// if
    
    //
    // perform column pivoted QR of M
    //

    auto  R = blas::matrix< value_t >( ncols, ncols );
    auto  P = std::vector< blas::int_t >( ncols, 0 );

    blas::qrp( M, R, P );

    auto  k = detail::trunc_rank( R, acc );
    
    //
    // restrict first k columns
    //

    // U = Q_k
    auto  Qk = blas::matrix< value_t >( M, blas::range::all, blas::range( 0, k-1 ) );
    auto  U  = blas::copy( Qk );

    // copy first k columns of R' to V, i.e., first k rows of R
    auto  Rk = blas::matrix< value_t >( R, blas::range( 0, k-1 ), blas::range::all );
    auto  TV = blas::matrix< value_t >( ncols, k );
    
    blas::copy( blas::adjoint( Rk ), TV );
    
    // then permute rows of TV (do P·R') and copy to V
    auto  V = blas::matrix< value_t >( ncols, k );
    
    for ( int i = 0; i < ncols; ++i )
    {
        auto  j    = P[i];
        auto  TV_i = TV.row( i );
        auto  V_j  = V.row( j );

        copy( TV_i, V_j );
    }// for

    return { std::move( U ), std::move( V ) };
}

//
// truncate low-rank matrix U·V' up to accuracy <acc>
//
template <typename T>
std::pair< blas::matrix< T >, blas::matrix< T > >
rrqr ( const blas::matrix< T > &  U,
       const blas::matrix< T > &  V,
       const accuracy &           acc )
{
    using  value_t = T;

    HLRCOMPRESS_ASSERT( U.ncols() == V.ncols() );

    const auto  nrows_U = idx_t( U.nrows() );
    const auto  nrows_V = idx_t( V.nrows() );
    const auto  in_rank = idx_t( V.ncols() );

    //
    // don't increase rank
    //

    if ( in_rank == 0 )
    {
        return { std::move( blas::matrix< value_t >( nrows_U, 0 ) ),
                 std::move( blas::matrix< value_t >( nrows_V, 0 ) ) };
    }// if

    //
    // if input rank is larger than maximal rank, use dense approximation
    //

    if ( in_rank > std::min( nrows_U, nrows_V ) )
    {
        auto  M = blas::prod( value_t(1), U, blas::adjoint(V) );

        return rrqr( M, acc );
    }// if
    else
    {
        // [ QV, RV ] = qr( V )
        auto  QV = blas::copy( V );
        auto  RV = blas::matrix< value_t >( in_rank, in_rank );

        blas::qr( QV, RV );

        // compute column-pivoted QR of U·RV'
        auto  QU = blas::prod( value_t(1), U, adjoint(RV) );
        auto  RU = blas::matrix< value_t >( in_rank, in_rank );
        auto  P  = std::vector< blas::int_t >( in_rank, 0 );

        blas::qrp( QU, RU, P );

        auto  out_rank = detail::trunc_rank( RU, acc );
        
        // U = QU(:,1:k)
        auto  Qk = blas::matrix< value_t >( QU, blas::range::all, blas::range( 0, out_rank-1 ) );
        auto  OU = blas::copy( Qk );
        
        // V = QV · P  (V' = P' · QV')
        auto  QV_P = blas::matrix< value_t >( nrows_V, in_rank );
        
        for ( int  i = 0; i < in_rank; ++i )
        {
            auto  j      = P[i];
            auto  QV_P_i = QV_P.column( i );
            auto  Q_j    = QV.column( j );

            blas::copy( Q_j, QV_P_i );
        }// for

        auto  Rk = blas::matrix< value_t >( RU, blas::range( 0, out_rank-1 ), blas::range( 0, in_rank-1 ) );
        auto  OV = blas::prod( value_t(1), QV_P, blas::adjoint( Rk ) );

        return { std::move( OU ), std::move( OV ) };
    }// else
}

//////////////////////////////////////////////////////////////////////
//
// provide above functions as functor
//
//////////////////////////////////////////////////////////////////////

struct RRQR
{
    //
    // matrix approximation routines
    //
    
    template < typename value_t >
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( blas::matrix< value_t > &  M,
                  const accuracy &           acc ) const
    {
        return rrqr( M, acc );
    }

    template < typename value_t >
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const blas::matrix< value_t > &  U,
                  const blas::matrix< value_t > &  V,
                  const accuracy &                 acc ) const
    {
        return rrqr( U, V, acc );
    }
};

}// namespace hlrcompress

#endif // __HLRCOMPRESS_APPROX_RRQR_HH
