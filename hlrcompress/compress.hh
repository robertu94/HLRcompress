#ifndef __HLRCOMPRESS_COMPRESS_HH
#define __HLRCOMPRESS_COMPRESS_HH
//
// Project     : HLRcompress
// Module      : compress
// Description : main compression function
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <memory>

#include <hlrcompress/config.h>

#if USE_TBB == 1
#  include <tbb/parallel_for.h>
#  include <tbb/blocked_range2d.h>
#endif

#include <hlrcompress/hlr/lowrank_block.hh>
#include <hlrcompress/hlr/dense_block.hh>
#include <hlrcompress/hlr/structured_block.hh>
#include <hlrcompress/approx/accuracy.hh>
#include <hlrcompress/misc/tensor.hh>

namespace hlrcompress
{

//
// build hierarchical low-rank format from given dense matrix without reording rows/columns
// starting lowrank approximation at blocks of size ntile × ntile and then trying
// to agglomorate low-rank blocks up to the root
//
namespace detail
{

template < typename value_t,
           typename approx_t >
std::unique_ptr< block< value_t > >
compress ( const indexset &                 rowis,
           const indexset &                 colis,
           const blas::matrix< value_t > &  D,
           const accuracy &                 acc,
           const approx_t &                 approx,
           const size_t                     ntile,
           const zconfig_t *                zconf = nullptr )
{
    if ( std::min( D.nrows(), D.ncols() ) <= ntile )
    {
        //
        // build leaf
        //
        // Apply low-rank approximation and compare memory consumption
        // with dense representation. If low-rank format uses less memory
        // the leaf is represented as low-rank (considered admissible).
        // Otherwise a dense representation is used.
        //

        if ( ! acc.is_exact() )
        {
            auto  Dc       = blas::copy( D );  // do not modify D (!)
            auto  [ U, V ] = approx( Dc, acc( rowis, colis ) );
            
            if ( U.byte_size() + V.byte_size() < Dc.byte_size() )
            {
                return std::make_unique< lowrank_block< value_t > >( rowis, colis, std::move( U ), std::move( V ) );
            }// if
        }// if

        return std::make_unique< dense_block< value_t > >( rowis, colis, std::move( blas::copy( D ) ) );
    }// if
    else
    {
        //
        // Recursion
        //
        // If all sub blocks are low-rank, an agglomorated low-rank matrix of all sub-blocks
        // is constructed. If the memory of this low-rank matrix is smaller compared to the
        // combined memory of the sub-block, it is kept. Otherwise a block matrix with the
        // already constructed sub-blocks is created.
        //

        const auto  mid_row = ( rowis.first() + rowis.last() + 1 ) / 2;
        const auto  mid_col = ( colis.first() + colis.last() + 1 ) / 2;

        indexset    sub_rowis[2] = { indexset( rowis.first(), mid_row-1 ), indexset( mid_row, rowis.last() ) };
        indexset    sub_colis[2] = { indexset( colis.first(), mid_col-1 ), indexset( mid_col, colis.last() ) };
        auto        sub_D        = tensor2< std::unique_ptr< block< value_t > > >( 2, 2 );

        #if USE_TBB == 1
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, 2, 0, 2 ),
            [&,ntile] ( const auto &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        const auto  D_sub = D( sub_rowis[i] - rowis.first(),
                                               sub_colis[j] - colis.first() );
                        
                        sub_D(i,j) = compress( sub_rowis[i], sub_colis[j], D_sub, acc, approx, ntile, zconf );
                        
                        HLRCOMPRESS_ASSERT( sub_D(i,j).get() != nullptr );
                    }// for
                }// for
            } );

        #elif USE_OPENMP == 1

        #pragma omp taskgroup
        {
            #pragma omp taskloop collapse(2) default(shared)
            for ( uint  i = 0; i < 2; ++i )
            {
                for ( uint  j = 0; j < 2; ++j )
                {
                    const auto  D_sub = D( sub_rowis[i] - rowis.first(),
                                           sub_colis[j] - colis.first() );
                        
                    sub_D(i,j) = compress( sub_rowis[i], sub_colis[j], D_sub, acc, approx, ntile, zconf );
                        
                    HLRCOMPRESS_ASSERT( sub_D(i,j).get() != nullptr );
                }// for
            }// for
        }// omp taskgroup
        
        #else

        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                const auto  D_sub = D( sub_rowis[i] - rowis.first(),
                                       sub_colis[j] - colis.first() );
                
                sub_D(i,j) = compress( sub_rowis[i], sub_colis[j], D_sub, acc, approx, ntile, zconf );
                
                HLRCOMPRESS_ASSERT( sub_D(i,j).get() != nullptr );
            }// for
        }// for
        
        #endif

        bool  all_lowrank = true;
        bool  all_dense   = true;

        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                if ( ! sub_D(i,j)->is_lowrank() )
                    all_lowrank = false;
                
                if ( ! sub_D(i,j)->is_dense() )
                    all_dense = false;
            }// for
        }// for
        
        if ( all_lowrank )
        {
            //
            // construct larger lowrank matrix out of smaller sub blocks
            //

            // compute initial total rank
            uint  rank_sum = 0;

            for ( uint  i = 0; i < 2; ++i )
                for ( uint  j = 0; j < 2; ++j )
                    rank_sum += static_cast< lowrank_block< value_t > * >( sub_D(i,j).get() )->rank();

            // copy sub block data into global structure
            auto    U    = blas::matrix< value_t >( rowis.size(), rank_sum );
            auto    V    = blas::matrix< value_t >( colis.size(), rank_sum );
            auto    pos  = 0; // pointer to next free space in U/V
            size_t  smem = 0; // holds memory of sub blocks
            
            for ( uint  i = 0; i < 2; ++i )
            {
                for ( uint  j = 0; j < 2; ++j )
                {
                    auto  Rij   = static_cast< lowrank_block< value_t > * >( sub_D(i,j).get() );
                    auto  Uij   = Rij->U();
                    auto  Vij   = Rij->V();
                    auto  U_sub = U( sub_rowis[i] - rowis.first(), blas::range( pos, pos + Uij.ncols() ) );
                    auto  V_sub = V( sub_colis[j] - colis.first(), blas::range( pos, pos + Uij.ncols() ) );

                    blas::copy( Uij, U_sub );
                    blas::copy( Vij, V_sub );

                    pos  += Uij.ncols();
                    smem += Uij.byte_size() + Vij.byte_size();
                }// for
            }// for

            //
            // try to approximate again in lowrank format and use
            // approximation if it uses less memory 
            //
            
            auto  [ W, X ] = approx( U, V, acc( rowis, colis ) );

            if ( W.byte_size() + X.byte_size() < smem )
            {
                return std::make_unique< lowrank_block< value_t > >( rowis, colis, std::move( W ), std::move( X ) );
            }// if
        }// if

        //
        // always join dense blocks
        //
        
        if ( all_dense )
        {
            return std::make_unique< dense_block< value_t > >( rowis, colis, std::move( blas::copy( D ) ) );
        }// if
        
        //
        // either not all low-rank or memory gets larger: construct block matrix
        // also: finally compress with zfp
        //

        auto  B = std::make_unique< structured_block< value_t > >( rowis, colis );

        B->set_block_struct( 2, 2 );
        
        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                if ( zconf != nullptr )
                {
                    if ( ! sub_D(i,j)->is_structured() )
                        sub_D(i,j)->compress( *zconf );
                }// if
                
                B->set_sub_block( i, j, sub_D(i,j).release() );
            }// for
        }// for

        return B;
    }// else
}

}// namespace detail

template < typename value_t,
           typename approx_t >
std::unique_ptr< block< value_t > >
compress ( const blas::matrix< value_t > &  D,
           const accuracy &                 acc,
           const approx_t &                 approx,
           const size_t                     ntile,
           const zconfig_t *                zconf = nullptr )
{
    auto  M = std::unique_ptr< block< value_t > >();

    #if USE_TBB == 1
    
    M = std::move( detail::compress( indexset( 0, D.nrows()-1 ), indexset( 0, D.ncols()-1 ), D, acc, approx, ntile, zconf ) );

    #elif USE_OPENMP == 1

    #pragma omp parallel
    #pragma omp single
    #pragma omp task
    M = std::move( detail::compress( indexset( 0, D.nrows()-1 ), indexset( 0, D.ncols()-1 ), D, acc, approx, ntile, zconf ) );

    #else
    
    M = std::move( detail::compress( indexset( 0, D.nrows()-1 ), indexset( 0, D.ncols()-1 ), D, acc, approx, ntile, zconf ) );
    
    #endif

    HLRCOMPRESS_ASSERT( M.get() != nullptr );

    // handle ZFP compression for global lowrank/dense case
    if (( zconf != nullptr ) && ! M->is_structured() )
        M->compress( *zconf );

    return M;
}

}// namespace hlrcompress

#endif // __HLRCOMPRESS_COMPRESS_HH