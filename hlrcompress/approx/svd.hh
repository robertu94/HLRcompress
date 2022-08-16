#ifndef __HLRCOMPRESS_SVD_HH
#define __HLRCOMPRESS_SVD_HH
//
// Project     : HLRcompress
// Module      : approx/svd
// Description : low-rank approximation functions using SVD
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <utility>

#include <hlrcompress/blas/arith.hh>
#include <hlrcompress/misc/type_traits.hh>
#include <hlrcompress/misc/error.hh>

namespace hlrcompress
{

//
// return low-rank approximation of M with accuracy <acc>
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
svd ( blas::matrix< value_t > &  M,
      const accuracy &           acc )
{
    using  real_t = real_type_t< value_t >;

    //
    // perform SVD of M
    //

    const idx_t  nrows_M = idx_t( M.nrows() );
    const idx_t  ncols_M = idx_t( M.ncols() );
    const idx_t  mrc     = std::min( nrows_M, ncols_M );
    auto         S       = blas::vector< real_t >( mrc );
    auto         V       = blas::matrix< value_t >( ncols_M, mrc );

    blas::svd( M, S, V );
        
    // determine truncated rank based on singular values
    const idx_t  k = idx_t( acc.trunc_rank( S ) );

    //
    // u_i -> a_i, v_i -> b_i
    // scale smaller one of A and B by S
    //

    const auto  Uk = blas::matrix< value_t >( M, blas::range::all, blas::range( 0, k-1 ) );
    const auto  Vk = blas::matrix< value_t >( V, blas::range::all, blas::range( 0, k-1 ) );
    auto        A = blas::copy( Uk );
    auto        B = blas::copy( Vk );

    if ( nrows_M < ncols_M )
        prod_diag( A, S, k );
    else
        prod_diag( B, S, k );

    return { std::move( A ), std::move( B ) };
}

//
// truncate low-rank matrix U·V' up to accuracy <acc>
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
svd ( const blas::matrix< value_t > &  U,
      const blas::matrix< value_t > &  V,
      const accuracy &                 acc )
{
    using  real_t  = typename real_type< value_t >::type_t;

    HLRCOMPRESS_ASSERT( U.ncols() == V.ncols() );

    const idx_t  nrows_U = idx_t( U.nrows() );
    const idx_t  nrows_V = idx_t( V.nrows() );
    const idx_t  in_rank = idx_t( V.ncols() );

    //
    // don't increase rank
    //

    if ( in_rank == 0 )
        return { std::move( blas::matrix< value_t >( nrows_U, 0 ) ),
                 std::move( blas::matrix< value_t >( nrows_V, 0 ) ) };

    //
    // truncate given low-rank matrix
    //
    
    if ( in_rank >= std::min( nrows_U, nrows_V ) )
    {
        //
        // since rank is too large, build U = U·V^T and do full-SVD
        //
            
        auto  M    = blas::prod( value_t(1), U, adjoint(V) );
        auto  lacc = accuracy( acc );

        return svd( M, lacc );
    }// if
    else
    {
        //////////////////////////////////////////////////////////////
        //
        // QR-factorisation of U and V with explicit Q
        //

        auto  QU = blas::copy( U );
        auto  RU = blas::matrix< value_t >( in_rank, in_rank );
        
        blas::qr( QU, RU );
        
        auto  QV = blas::copy( V );
        auto  RV = std::move( blas::matrix< value_t >( in_rank, in_rank ) );
        
        blas::qr( QV, RV );

        //
        // R = R_U · upper_triangular(QV)^H = R_V^H
        //
        
        auto  R = blas::prod( value_t(1), RU, adjoint(RV) );
        
        //
        // SVD(R) = U S V^H
        //
            
        blas::vector< real_t >   Ss( in_rank );
        blas::matrix< value_t >  Us( std::move( R ) );  // reuse storage
        blas::matrix< value_t >  Vs( std::move( RV ) );
            
        blas::svd( Us, Ss, Vs );
        
        // determine truncated rank based on singular values
        const auto  orank = idx_t( acc.trunc_rank( Ss ) );

        //
        // only build new vectors, if rank is decreased
        //
        
        if ( orank < in_rank )
        {
            //
            // build new matrices U and V
            //

            const blas::range  in_rank_is( 0, in_rank-1 );
            const blas::range  orank_is( 0, orank-1 );

            // U := Q_U · U
            blas::matrix< value_t >  Urank( Us, in_rank_is, orank_is );
            
            // U := U·S
            blas::prod_diag( Urank, Ss, orank );

            auto  OU = blas::prod( value_t(1), QU, Urank );
            
            // V := Q_V · conj(V)
            blas::matrix< value_t >  Vrank( Vs, in_rank_is, orank_is );

            auto  OV = blas::prod( value_t(1), QV, Vrank );

            return { std::move( OU ), std::move( OV ) };
        }// if
        else
        {
            // rank has not changed, so return original matrices
            return { std::move( blas::copy( U ) ), std::move( blas::copy( V ) ) };
        }// else
    }// else
}

//////////////////////////////////////////////////////////////////////
//
// provide above functions as functor
//
//////////////////////////////////////////////////////////////////////

struct SVD
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
        return svd( M, acc );
    }

    template < typename value_t >
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const blas::matrix< value_t > &  U,
                  const blas::matrix< value_t > &  V,
                  const accuracy &                 acc ) const 
    {
        return svd( U, V, acc );
    }
};

}// namespace hlrcompress

#endif // __HLRCOMPRESS_SVD_HH
