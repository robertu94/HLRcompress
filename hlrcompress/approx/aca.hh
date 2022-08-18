#ifndef __HLRCOMPRESS_ACA_HH
#define __HLRCOMPRESS_ACA_HH
//
// Project     : HLRcompress
// Module      : approx/aca
// Description : low-rank approximation functions using ACA
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <vector>
#include <list>
#include <deque>
#include <utility>

#include <hlrcompress/blas/arith.hh>
#include <hlrcompress/approx/svd.hh>

namespace hlrcompress
{

//
// compute low-rank approximation of M using ACA with standard pivot search
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
aca  ( const blas::matrix< value_t > &  M,
       const accuracy &                 acc,
       const bool                       recompress = true )
{
    using  real_t = real_type_t< value_t >;

    // value considered zero to avoid division by small values
    static constexpr real_t  zero_val = std::numeric_limits< real_t >::epsilon() * std::numeric_limits< real_t >::epsilon();

    // matrix data
    const auto  nrows_M  = M.nrows();
    const auto  ncols_M  = M.ncols();
    const auto  min_dim  = std::min( nrows_M, ncols_M );
    
    // maximal rank either defined by accuracy or dimension of matrix
    const auto  max_rank = ( acc.has_max_rank() ? std::min( min_dim, acc.max_rank() ) : min_dim );
    
    // precision defined by accuracy or by machine precision
    // (to be corrected by matrix norm)
    real_t      rel_eps  = acc.rel_eps();
    real_t      abs_eps  = acc.abs_eps();
    
    // approximation of |M|
    real_t      norm_M   = real_t(0);

    // to remember pivot columns/rows
    int                  next_col = 0;
    std::vector< bool >  used_rows( nrows_M, false );
    std::vector< bool >  used_cols( ncols_M, false );
    
    // low-rank approximation
    std::deque< blas::vector< value_t > >  U, V;
    
    for ( uint  i = 0; i < max_rank; ++i )
    {
        //
        // determine pivot pair
        //

        const auto  pivot_col = next_col;
        
        auto  column = blas::copy( M.column( pivot_col ) );
        
        for ( uint  l = 0; l < U.size(); ++l )
            blas::add( -std::conj( V[l]( pivot_col ) ), U[l], column );
        
        const auto  pivot_row = blas::max_idx( column );
        const auto  max_val   = column( pivot_row );

        // stop and signal no pivot found if remainder is "zero"
        if ( std::abs( max_val ) <= zero_val )
            return { -1, -1, blas::vector< value_t >(), blas::vector< value_t >() };

        // scale <col> by inverse of maximal element in u
        blas::scale( value_t(1) / max_val, column );
        
        used_rows[ pivot_row ] = true;
        
        auto  row = blas::copy( M.row( pivot_row ) );

        // stored as column, hence conjugate
        blas::conj( row );
        
        for ( uint  l = 0; l < U.size(); ++l )
            blas::add( -std::conj( U[l]( pivot_row ) ), V[l], row );

        //
        // for next column, look for maximal element in computed row
        //

        real_t  max_v = real_t(0);
        int     max_j = -1;

        for ( size_t  j = 0; j < row.length(); ++j )
        {
            if ( ! used_cols[ j ] )
            {
                if ( std::abs( row(j) ) > max_v )
                {
                    max_v = std::abs( row(j) );
                    max_j = j;
                }// if
            }// if
        }// for
                
        next_col = max_j;

        if (( pivot_row == -1 ) || ( pivot_col == -1 ))
            break;
        
        //
        // test convergence by comparing |u_i·v_i'| (approx. for remainder)
        // with |M| ≅ |U·V'|
        //
        
        const auto  norm_i = blas::norm2( column ) * blas::norm2( row );

        if (( norm_i < rel_eps * norm_M ) || ( norm_i < abs_eps ))
        {
            U.push_back( std::move( column ) );
            V.push_back( std::move( row ) );
            break;
        }// if

        //
        // update approx. of |M|
        //
        //   |U(:,1:k)·V(:,1:k)'|² = ∑_r=1:k ∑_l=1:k u_r'·u_l  v_r'·v_l
        //                         = |U(:,1:k-1)·V(:,1:k-1)|²
        //                           + ∑_l=1:k-1 u_k'·u_l  v_k'·v_l
        //                           + ∑_l=1:k-1 u_l'·u_k  v_l'·v_k
        //                           + u_k·u_k v_k·v_k
        //

        value_t  upd = norm_i*norm_i;
        
        for ( uint  l = 0; l < U.size(); ++l )
            upd += ( blas::dot( U[l],   column ) * blas::dot( V[l], row  ) +
                     blas::dot( column, U[l]   ) * blas::dot( row,  V[l] ) );

        norm_M = std::sqrt( norm_M * norm_M + std::abs( upd ) );
        
        //
        // and store new vectors
        //

        U.push_back( std::move( column ) );
        V.push_back( std::move( row ) );
    }// while
    
    //
    // copy to matrices and return
    //
    
    blas::matrix< value_t >  MU( nrows_M, U.size() );
    blas::matrix< value_t >  MV( ncols_M, V.size() );

    for ( uint  l = 0; l < U.size(); ++l )
    {
        auto  u_l = MU.column( l );
        auto  v_l = MV.column( l );

        blas::copy( U[l], u_l );
        blas::copy( V[l], v_l );
    }// for
    
    if ( recompress )
        return svd( MU, MV, acc );
    else
        return { std::move( MU ), std::move( MV ) };
}

//
// compute low-rank approximation of M using ACA with full pivot search
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
aca_full  ( blas::matrix< value_t > &  M,
            const accuracy &           acc,
            const bool                 recompress = true )
{
    using  real_t = real_type_t< value_t >;

    // value considered zero to avoid division by small values
    static constexpr real_t  zero_val = std::numeric_limits< real_t >::epsilon() * std::numeric_limits< real_t >::epsilon();

    // matrix data
    const auto  nrows_M  = M.nrows();
    const auto  ncols_M  = M.ncols();
    const auto  min_dim  = std::min( nrows_M, ncols_M );
    
    // maximal rank either defined by accuracy or dimension of matrix
    const auto  max_rank = ( acc.has_max_rank() ? std::min( min_dim, acc.max_rank() ) : min_dim );
    
    // precision defined by accuracy or by machine precision
    // (to be corrected by matrix norm)
    real_t      rel_eps  = acc.rel_eps();
    real_t      abs_eps  = acc.abs_eps();
    
    real_t      norm_M   = blas::normF( M );

    // to remember pivot columns/rows
    int                  next_col = 0;
    std::vector< bool >  used_rows( nrows_M, false );
    std::vector< bool >  used_cols( ncols_M, false );
    
    // low-rank approximation
    std::deque< blas::vector< value_t > >  U, V;
    
    for ( uint  i = 0; i < max_rank; ++i )
    {
        //
        // determine pivot pair
        //

        value_t  max_val   = value_t(0);
        int      pivot_col = 0;
        int      pivot_row = 0;
        
        for ( size_t  j = 0; j < ncols_M; ++j )
        {
            const auto  column  = M.column( j );
            const auto  max_row = blas::max_idx( column );
            const auto  val     = column( max_row );

            if ( std::abs( val ) > std::abs( max_val ) )
            {
                pivot_col = j;
                pivot_row = max_row;
                max_val   = val;
            }// if
        }// for

        // stop and signal no pivot found if remainder is "zero"
        if ( std::abs( max_val ) <= zero_val )
            break;
        
        auto  column = blas::copy( M.column( pivot_col ) );
        
        // scale <col> by inverse of maximal element in u
        blas::scale( value_t(1) / max_val, column );
        
        auto  row = blas::copy( M.row( pivot_row ) );

        // stored as column, hence conjugate
        blas::conj( row );
        
        //
        // test convergence by comparing |u_i·v_i'| (approx. for remainder)
        // with |M| ≅ |U·V'|
        //

        blas::add_r1( value_t(-1), column, row, M );
        
        const auto  norm_i = blas::normF( M );

        if (( norm_i < rel_eps * norm_M ) || ( norm_i < abs_eps ))
        {
            U.push_back( std::move( column ) );
            V.push_back( std::move( row ) );
            break;
        }// if

        //
        // and store new vectors
        //

        U.push_back( std::move( column ) );
        V.push_back( std::move( row ) );
    }// while
    
    //
    // copy to matrices and return
    //
    
    blas::matrix< value_t >  MU( nrows_M, U.size() );
    blas::matrix< value_t >  MV( ncols_M, V.size() );

    for ( uint  l = 0; l < U.size(); ++l )
    {
        auto  u_l = MU.column( l );
        auto  v_l = MV.column( l );

        blas::copy( U[l], u_l );
        blas::copy( V[l], v_l );
    }// for

    if ( recompress )
        return svd( MU, MV, acc );
    else
        return { std::move( MU ), std::move( MV ) };
}

//
// truncate low-rank matrix U·V' up to accuracy <acc>
//
template < typename value_t >
std::pair< blas::matrix< value_t >,
           blas::matrix< value_t > >
aca_full ( const blas::matrix< value_t > &  U,
           const blas::matrix< value_t > &  V,
           const accuracy &                 acc )
{
    using  real_t = real_type_t< value_t >;

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
    
    if ( in_rank >= std::min( nrows_U, nrows_V ) / 2 )
    {
        //
        // since rank is too large, build U = U·V^T and do full-SVD
        //
            
        auto  M    = blas::prod( value_t(1), U, adjoint(V) );
        auto  lacc = accuracy( acc );

        return aca_full( M, lacc );
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
        // low-rank approximation of R
        //
            
        auto  [ Us, Vs ] = aca_full( R, acc );
        
        const auto  out_rank = Us.ncols();

        //
        // only build new vectors, if rank is decreased
        //
        
        if ( out_rank < in_rank )
        {
            auto  OU = blas::prod( value_t(1), QU, Us );
            auto  OV = blas::prod( value_t(1), QV, Vs );

            return { std::move( OU ), std::move( OV ) };
        }// if
        else
        {
            // rank has not changed, so return original matrices
            return { std::move( blas::copy( U ) ), std::move( blas::copy( V ) ) };
        }// else
    }// else
}

}// namespace hlrcompress

#endif // __HLRCOMPRESS_ACA_HH
