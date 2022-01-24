#ifndef __HLRCOMPRESS_ERROR_HH
#define __HLRCOMPRESS_ERROR_HH
//
// Project     : HLRcompress
// Module      : hlr/error
// Description : error computation functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hlrcompress/config.h>

#if USE_TBB == 1
#  include <mutex>
#  include <tbb/parallel_for.h>
#  include <tbb/blocked_range2d.h>
#endif

#include <hlrcompress/hlr/structured_block.hh>
#include <hlrcompress/hlr/lowrank_block.hh>
#include <hlrcompress/hlr/dense_block.hh>
#include <hlrcompress/misc/type_traits.hh>

namespace hlrcompress
{

namespace detail
{

template < typename value_t >
real_type_t< value_t >
sqerror_fro ( const blas::matrix< value_t > &  D,
              const block< value_t > &         Z )
{
    using  real_t = real_type_t< value_t >;
    
    if ( Z.is_structured() )
    {
        auto    BZ  = static_cast< const structured_block< value_t > * >( & Z );
        real_t  err = real_t(0);

        #if USE_TBB == 1

        auto  mtx = std::mutex();
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, 2, 0, 2 ),
            [&,BZ] ( const auto &  r )
            {
                auto  loc_err = real_t(0);
                
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        loc_err += sqerror_fro< value_t >( D, BZ->sub_block( i, j ) );
                    }// for
                }// for

                {
                    auto  lock = std::scoped_lock( mtx );
                    
                    err += loc_err;
                }
            } );

        #elif USE_OPENMP == 1

        #pragma omp taskgroup
        {
            #pragma omp taskloop collapse(2) default(shared) reduction(+: err)
            for ( uint  i = 0; i < 2; ++i )
                for ( uint  j = 0; j < 2; ++j )
                    err += sqerror_fro( D, BZ->sub_block( i, j ) );
        }// omp taskgroup
        
        #else

        for ( uint  i = 0; i < BZ->nblock_rows(); ++i )
            for ( uint  j = 0; j < BZ->nblock_cols(); ++j )
                err += sqerror_fro( D, BZ->sub_block( i, j ) );

        #endif

        return err;
    }// if
    else if ( Z.is_lowrank() )
    {
        auto  RZ    = static_cast< const lowrank_block< value_t > * >( & Z );
        auto  D_sub = blas::matrix< value_t >( D, RZ->row_is(), RZ->col_is() );
        auto  D_cpy = blas::copy( D_sub );

        blas::prod( value_t(-1), RZ->U(), blas::adjoint( RZ->V() ), value_t(1), D_cpy );

        auto  err   = blas::norm_F( D_cpy );

        return err * err;
    }// if
    else if ( Z.is_dense() )
    {
        auto  DZ    = static_cast< const dense_block< value_t > * >( & Z );
        auto  D_sub = blas::matrix< value_t >( D, DZ->row_is(), DZ->col_is() );
        auto  D_cpy = blas::copy( D_sub );

        blas::add( value_t(-1), DZ->M(), D_cpy );

        auto  err   = blas::norm_F( D_cpy );

        return err * err;
    }// if
    else
        HLRCOMPRESS_ERROR( "invalid compression type" );

    return real_t(0);
}

}// namespace detail

template < typename value_t >
real_type_t< value_t >
error_fro ( const blas::matrix< value_t > &  D,
            const block< value_t > &         Z )
{
    auto  sqerr = detail::sqerror_fro( D, Z );

    if constexpr( is_complex_type_v< value_t > )
    {
        return std::sqrt( std::real( sqerr ) );
    }// if
    else
    {
        return std::sqrt( sqerr );
    }// else
}
    
}// namespace hlrcompress

#endif // __HLRCOMPRESS_ERROR_HH
