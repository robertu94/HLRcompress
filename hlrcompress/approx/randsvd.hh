#ifndef __HLRCOMPRESS_APPROX_RANDSVD_HH
#define __HLRCOMPRESS_APPROX_RANDSVD_HH
//
// Project     : HLRcompress
// Module      : approx/randsvd
// Description : low-rank approximation functions using randomized SVD
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <list>
#include <random>

#include <hlrcompress/blas/arith.hh>
#include <hlrcompress/misc/type_traits.hh>
#include <hlrcompress/misc/error.hh>

namespace hlrcompress
{

namespace detail
{

//
// compute basis for column space (range) of M
//
template < typename value_t >
blas::matrix< value_t >
rand_column_basis ( const blas::matrix< value_t > &  M,
                    const accuracy &                 acc,
                    const uint                       block_size,
                    const uint                       power_steps,
                    const uint                       oversampling )
{
    using  real_t = real_type_t< value_t >;
        
    auto  rd        = std::random_device{};
    auto  generator = std::mt19937{ rd() };
    auto  distr     = std::normal_distribution<>{ 0, 1 };
    auto  rand_norm = [&] () { return distr( generator ); };
    
    const auto  nrows_M = M.nrows();
    const auto  ncols_M = M.ncols();
    real_t      norm_M  = real_t(0);
    const auto  rel_eps = acc.rel_eps();
    const auto  abs_eps = acc.abs_eps();
    const uint  bsize   = std::min< uint >( block_size, std::min< uint >( nrows_M, ncols_M ) );
    const uint  nblocks = std::min< uint >( nrows_M, ncols_M ) / bsize;
    auto        Qs      = std::list< blas::matrix< value_t > >();
    auto        T_i     = blas::matrix< value_t >( ncols_M, bsize );
    auto        QhQi    = blas::matrix< value_t >( bsize,   bsize );
    auto        TQ_i    = blas::matrix< value_t >( nrows_M, bsize );
    auto        R       = blas::matrix< value_t >( bsize,   bsize );
    auto        MtQ     = blas::matrix< value_t >( ncols_M, bsize );

    for ( uint  i = 0; i < nblocks; ++i )
    {
        //
        // draw random matrix and compute approximation of remainder M - Σ_j Q_j·Q_j'·M
        //
            
        for ( size_t  j = 0; j < T_i.ncols(); ++j )
            for ( size_t  i = 0; i < T_i.nrows(); ++i )
                T_i(i,j) = rand_norm();
            
        auto  Q_i = blas::prod( value_t(1), M, T_i );

        // subtract previous Q_j
        if ( ! Qs.empty() )
        {
            blas::copy( Q_i, TQ_i );
            
            for ( auto  Q_j : Qs )
            {
                blas::prod( value_t(1), blas::adjoint( Q_j ), TQ_i, value_t(0), QhQi );
                blas::prod( value_t(-1), Q_j, QhQi, value_t(1), Q_i );
            }// for
        }// if

        //
        // compute norm of remainder and update norm(M)
        //
            
        real_t  norm_Qi = real_t(0);

        for ( uint  j = 0; j < bsize; ++j )
        {
            const auto  Qi_j = Q_i.column( j );

            norm_Qi = std::max( norm_Qi, blas::norm2( Qi_j ) );
        }// for

        norm_M = std::sqrt( norm_M * norm_M + norm_Qi * norm_Qi );

        //
        // power iteration
        //
            
        blas::qr( Q_i, R );
            
        if ( power_steps > 0 )
        {
            for ( uint  j = 0; j < power_steps; ++j )
            {
                blas::prod( value_t(1), blas::adjoint( M ), Q_i, value_t(0), MtQ );
                blas::qr( MtQ, R );
                    
                blas::prod( value_t(1), M, MtQ, value_t(0), Q_i );
                blas::qr( Q_i, R );
            }// for
        }// if
            
        //
        // project Q_i away from previous Q_j
        //
        //    Q_i = Q_i - [ Q_0 .. Q_i-1 ] [ Q_0 .. Q_i-1 ]^H Q_i = Q_i - Σ_j=0^i-1 Q_j Q_j^H Q_i
        //
                
        if ( i > 0 )
        {
            blas::copy( Q_i, TQ_i );
                
            for ( const auto &  Q_j : Qs )
            {
                blas::prod( value_t(1), blas::adjoint(Q_j), TQ_i, value_t(0), QhQi );
                blas::prod( value_t(-1), Q_j, QhQi, value_t(1), Q_i );
            }// for
                
            blas::qr( Q_i, R );
        }// if
            
        //
        // M = M - Q_i Q_i^t M
        //

        Qs.push_back( std::move( Q_i ) );
            
        if (( norm_Qi <= abs_eps ) || (( norm_Qi ) <= rel_eps * norm_M ))
            break;
    }// for
        
    //
    // collect Q_i's into final result
    //

    auto   Q   = blas::matrix< value_t >( nrows_M, Qs.size() * bsize );
    idx_t  pos = 0;

    for ( const auto &  Q_i : Qs )
    {
        auto  Q_sub = blas::matrix< value_t >( Q, blas::range::all, blas::range( pos, pos+bsize-1 ) );

        blas::copy( Q_i, Q_sub );
        pos += bsize;
    }// for

    return Q;
}

//
// computes column basis for U·V'
// - slightly faster than general version
//
template < typename value_t >
blas::matrix< value_t >
rand_column_basis ( const blas::matrix< value_t > &  U,
                    const blas::matrix< value_t > &  V,
                    const accuracy &                 acc,
                    const uint                       block_size,
                    const uint                       power_steps,
                    const uint                       oversampling )
{
    const idx_t  nrows = idx_t( U.nrows() );
    const idx_t  ncols = idx_t( V.nrows() );
    const idx_t  rank  = idx_t( U.ncols() );
    
    auto  rd        = std::random_device{};
    auto  generator = std::mt19937{ rd() };
    auto  distr     = std::normal_distribution<>{ 0, 1 };
    auto  rand_norm = [&] () { return distr( generator ); };
    
    auto        Uc      = blas::copy( U ); // need copy to be modified below
    const auto  norm_M  = blas::norm_F( Uc, V );
    const auto  rel_eps = acc.rel_eps();
    const auto  abs_eps = acc.abs_eps();
    const uint  bsize   = std::min< uint >( block_size, std::min< uint >( nrows, ncols ) );
    const uint  nblocks = std::min( nrows, ncols ) / bsize;
    auto        Qs      = std::list< blas::matrix< value_t > >();
    auto        T_i     = blas::matrix< value_t >( ncols, bsize );
    auto        VtT     = blas::matrix< value_t >( rank,  bsize );
    auto        TQ_i    = blas::matrix< value_t >( nrows, bsize );
    auto        UtQ     = blas::matrix< value_t >( rank,  bsize );
    auto        VUtQ    = blas::matrix< value_t >( ncols, bsize );
    auto        R       = blas::matrix< value_t >( bsize, bsize );
    auto        QjtQi   = blas::matrix< value_t >( bsize, bsize );
    auto        QtA     = blas::matrix< value_t >( bsize, rank );

    for ( uint  i = 0; i < nblocks; ++i )
    {
        for ( size_t  j = 0; j < T_i.ncols(); ++j )
            for ( size_t  i = 0; i < T_i.nrows(); ++i )
                T_i(i,j) = rand_norm();
        
        blas::prod( value_t(1), blas::adjoint(V), T_i, value_t(0), VtT );
            
        auto  Q_i = blas::prod( value_t(1), Uc, VtT );

        //
        // power iteration
        //
            
        blas::qr( Q_i, R );
            
        if ( power_steps > 0 )
        {
            for ( uint  j = 0; j < power_steps; ++j )
            {
                blas::prod( value_t(1), blas::adjoint(Uc), Q_i, value_t(0), UtQ );
                blas::prod( value_t(1), V, UtQ, value_t(0), VUtQ );
                blas::qr( VUtQ, R );
                    
                blas::prod( value_t(1), blas::adjoint(V), VUtQ, value_t(0), UtQ );
                blas::prod( value_t(1), Uc, UtQ, value_t(0), Q_i );
                blas::qr( Q_i, R );
            }// for
        }// if
            
        //
        // project Q_i away from previous Q_j
        //
                
        if ( i > 0 )
        {
            blas::copy( Q_i, TQ_i );
                
            for ( const auto &  Q_j : Qs )
            {
                blas::prod( value_t(1), blas::adjoint(Q_j), TQ_i, value_t(0), QjtQi );
                blas::prod( value_t(-1), Q_j, QjtQi, value_t(1), Q_i );
            }// for
                
            blas::qr( Q_i, R );
        }// if

        //
        // M = M - Q_i Q_i^T M = U·V^H - Q_i Q_i^T U·V^H = (U - Q_i Q_i^T U) V^H
        //

        blas::prod( value_t(1), blas::adjoint(Q_i), Uc, value_t(0), QtA );
        blas::prod( value_t(-1), Q_i, QtA, value_t(1), Uc );
            
        const auto  norm_Qi = blas::norm_F( Uc, V );

        Qs.push_back( std::move( Q_i ) );
            
        if (( norm_Qi < abs_eps ) || ( norm_Qi <= rel_eps * norm_M ))
            break;
    }// for

    //
    // collect Q_i's into final result
    //

    auto   Q   = blas::matrix< value_t >( nrows, Qs.size() * bsize );
    idx_t  pos = 0;

    for ( const auto &  Q_i : Qs )
    {
        auto  Q_sub = blas::matrix< value_t >( Q, blas::range::all, blas::range( pos * bsize, (pos+1)*bsize - 1 ) );

        blas::copy( Q_i, Q_sub );
        ++pos;
    }// for

    return Q;
}

}// namespace detail

//
// return low-rank approximation of M with accuracy <acc>
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
randsvd ( const blas::matrix< value_t > &  M,
          const accuracy &                 acc,
          const uint                       power_steps,
          const uint                       oversampling )
{
    using  real_t = real_type_t< value_t >;

    const auto  nrows_M = M.nrows();
    const auto  ncols_M = M.ncols();

    // compute column basis
    auto  Q   = detail::rand_column_basis( M, acc, 4, power_steps, oversampling );
    auto  k   = Q.ncols();

    // B = Q^H · M  or B^H = M^H · Q
    auto  BT  = blas::prod( value_t(1), adjoint( M ), Q );
    auto  R_B = blas::matrix< value_t >( k, k );
    auto  V   = blas::matrix< value_t >( k, k );
    auto  S   = blas::vector< real_t >( k );

    // B^T = Q_B R_B  (Q_B overwrites B)
    blas::qr( BT, R_B );

    // R_B = U·S·V^H
    blas::svd( R_B, S, V );

    // determine truncated rank based on singular values
    k = idx_t( acc.trunc_rank( S ) );

    // A = Y · V_k, B = B^T · U_k
    auto  Uk = blas::matrix< value_t >( R_B, blas::range::all, blas::range( 0, k-1 ) );
    auto  Vk = blas::matrix< value_t >( V,   blas::range::all, blas::range( 0, k-1 ) );
    
    auto  OU = blas::prod( value_t(1), Q,  Vk );
    auto  OV = blas::prod( value_t(1), BT, Uk );

    if ( nrows_M < ncols_M )
        blas::prod_diag( OU, S, k );
    else
        blas::prod_diag( OV, S, k );

    return { std::move( OU ), std::move( OV ) };
}

//
// truncate low-rank matrix U·V' up to accuracy <acc>
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
randsvd ( const blas::matrix< value_t > &  U,
          const blas::matrix< value_t > &  V,
          const accuracy &                 acc,
          const uint                       power_steps,
          const uint                       oversampling )
{
    using  real_t  = real_type_t< value_t >;

    HLRCOMPRESS_ASSERT( U.ncols() == V.ncols() );

    const idx_t  nrows_U = idx_t( U.nrows() );
    const idx_t  nrows_V = idx_t( V.nrows() );
    const idx_t  in_rank = idx_t( V.ncols() );

    //
    // don't increase rank
    //

    if ( in_rank == 0 )
    {
        return { std::move( blas::matrix< value_t >( nrows_U, 0 ) ),
                 std::move( blas::matrix< value_t >( nrows_V, 0 ) ) };
    }// if

    //
    // if k is bigger than the possible rank,
    // we create a dense-matrix and do truncation
    // via full SVD
    //

    if ( in_rank >= std::min( nrows_U, nrows_V ) )
    {
        auto  M = blas::prod( value_t(1), U, blas::adjoint(V) );

        return randsvd( M, acc, power_steps, oversampling );
    }// if
    else
    {
        //
        // compute column basis
        //

        auto  Q      = detail::rand_column_basis( U, V, acc, 4, power_steps, oversampling );
        auto  k_base = idx_t(Q.ncols());

        // Q^H · U · V^H  = (V·U^H·Q)^H
        auto  UtQ    = blas::prod( value_t(1), blas::adjoint(U), Q );
        auto  VUtQ   = blas::prod( value_t(1), V, UtQ );

        auto  U_svd  = blas::matrix< value_t >( k_base, k_base );
        auto  V_svd  = blas::matrix< value_t >( k_base, k_base );
        auto  S      = blas::vector< real_t >( k_base );

        // (V·U^H·Q)^H = Q_B R
        blas::qr( VUtQ, U_svd );
        
        // R_V = U·S·V^H
        svd( U_svd, S, V_svd );
        
        // determine truncated rank based on singular values
        auto  out_rank = idx_t( acc.trunc_rank( S ) );

        // A = Y · V_k, B = B^T · U_k
        auto  Uk = blas::matrix< value_t >( U_svd, blas::range::all, blas::range( 0, out_rank-1 ) );
        auto  Vk = blas::matrix< value_t >( V_svd, blas::range::all, blas::range( 0, out_rank-1 ) );

        auto  OU = blas::prod( value_t(1), Q,    Vk );
        auto  OV = blas::prod( value_t(1), VUtQ, Uk );

        if ( nrows_U < nrows_V )
            blas::prod_diag( OU, S, out_rank );
        else
            blas::prod_diag( OV, S, out_rank );

        return { std::move( OU ), std::move( OV ) };
    }// else
}

//////////////////////////////////////////////////////////////////////
//
// provide above functions as functor
//
//////////////////////////////////////////////////////////////////////

struct RandSVD
{
    // number of steps in power iteration during construction of column basis
    const uint   power_steps  = 0;

    // oversampling parameter
    const uint   oversampling = 0;

    //
    // matrix approximation routines
    //
    
    template < typename value_t >
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( blas::matrix< value_t > &  M,
                  const accuracy &           acc ) const
    {
        return randsvd( M, acc, power_steps, oversampling );
    }

    template < typename value_t >
    std::pair< blas::matrix< value_t >,
               blas::matrix< value_t > >
    operator () ( const blas::matrix< value_t > &  U,
                  const blas::matrix< value_t > &  V,
                  const accuracy &                 acc ) const
    {
        auto  Uc = blas::copy( U );
        auto  Vc = blas::copy( V );
        
        return randsvd( Uc, Vc, acc, power_steps, oversampling );
    }
};

}// namespace hlrcompress

#endif // __HLRCOMPRESS_APPROX_RANDSVD_HH
