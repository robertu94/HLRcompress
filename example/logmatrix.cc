//
// Project     : HLRcompress
// Module      : compress
// Description : compression example
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <cstdio>
#include <cmath>
#include <chrono>
#include <iomanip>

#include <hlrcompress/config.h>

#if HLRCOMPRESS_USE_TBB == 1
#  include <tbb/parallel_for.h>
#  include <tbb/blocked_range2d.h>
#endif

#include <hlrcompress/compress.hh>
#include <hlrcompress/compress_cuda.hh>
#include <hlrcompress/approx/svd.hh>
#include <hlrcompress/approx/rrqr.hh>
#include <hlrcompress/approx/randsvd.hh>
#include <hlrcompress/hlr/error.hh>

using namespace hlrcompress;

template < typename value_t >
blas::matrix< value_t >
gen_matrix_log ( const size_t  n )
{
    constexpr double  pi = 3.14159265358979323846;
    const     double  h  = 2 * pi / value_t(n);
    auto              M  = blas::matrix< value_t >( n, n );

    #if HLRCOMPRESS_USE_TBB == 1
    
    ::tbb::parallel_for(
        ::tbb::blocked_range2d< size_t >( 0, n, 1024,
                                          0, n, 1024 ),
        [&,h] ( const auto &  r )
        {
            auto  rofs  = r.rows().begin();
            auto  cofs  = r.cols().begin();
            auto  nrows = r.rows().end() - r.rows().begin();
            auto  ncols = r.cols().end() - r.cols().begin();

            auto  T = blas::matrix< double >( nrows, ncols );
            
            for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
            {
                const double  x2[2] = { std::sin(j*h), std::cos(j*h) };
        
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    const double  x1[2] = { std::sin(i*h), std::cos(i*h) };
                    const auto    d0    = x1[0] - x2[0];
                    const auto    d1    = x1[1] - x2[1];
                    const double  dist2 = d0*d0 + d1*d1;

                    if ( dist2 < 1e-12 )
                        T(i-rofs,j-cofs) = 0.0;
                    else
                        T(i-rofs,j-cofs) = std::log( std::sqrt(dist2) );
                }// for
            }// for

            auto  M_ij = blas::matrix< double >( M,
                                                 blas::range( rofs, rofs+nrows-1 ),
                                                 blas::range( cofs, cofs+ncols-1 ) );

            blas::copy( T, M_ij );
        } );

    #else

    #pragma omp parallel for collapse(2) schedule(static,1024)
    for ( size_t  i = 0; i < n; ++i )
    {
        for ( size_t  j = 0; j < n; ++j )
        {
            const double  x1[2] = { std::sin(i*h), std::cos(i*h) };
            const double  x2[2] = { std::sin(j*h), std::cos(j*h) };
            const auto    d0    = x1[0] - x2[0];
            const auto    d1    = x1[1] - x2[1];
            const double  dist2 = d0*d0 + d1*d1;

            if ( dist2 < 1e-12 )
                M(i,j) = 0.0;
            else
                M(i,j) = std::log( std::sqrt(dist2) );
        }// for
    }// for
    
    #endif

    return M;
}

int
main ( int      argc,
       char **  argv )
{
    const auto  n     = ( argc > 1 ? atoi( argv[1] ) : 128 );
    auto        M     = gen_matrix_log< double >( n );
    const auto  acc   = ( argc > 2 ? atof( argv[2] ) : 1e-4 );
    const auto  ntile = ( argc > 3 ? atoi( argv[3] ) : 32 );
    auto        apx   = SVD();

    // #if HLRCOMPRESS_USE_ZFP == 1
    // const auto  rate  = ( argc > 4 ? atoi( argv[4] ) : 32 );
    // auto        zconf = std::make_unique< zconfig_t >( zfp_config_rate( rate, false ) );
    // #else
    // auto        zconf = std::unique_ptr< zconfig_t >();
    // #endif

    auto        tic   = std::chrono::high_resolution_clock::now();

    #if HLRCOMPRESS_USE_CUDA == 1
    auto        zM    = compress_cuda( M, acc, apx, ntile );
    #else
    auto        zM    = compress( M, acc, apx, ntile );
    #endif
    
    auto        toc   = std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - tic );

    std::cout << "runtime:      " << toc.count() / 1e6 << " s" << std::endl;
    
    const auto  bs_M  = M.byte_size();
    const auto  bs_zM = zM->byte_size();

    std::cout << "storage " << std::endl
              << "  original:   " << bs_M << std::endl
              << "  compressed: " << bs_zM << " / " << ( 100.0 * double(bs_zM) / double(bs_M) ) << "%" << std::endl;

    // needs to be uncompressed for error computation (for now)
    zM->uncompress();
    
    auto  norm_M = blas::norm_F( M );
    
    std::cout << "error:        " << std::setprecision(4) << std::scientific << error_fro( M, *zM ) / norm_M << std::endl;
    
    return 0;
}
