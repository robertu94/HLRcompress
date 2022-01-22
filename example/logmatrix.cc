//
// Project     : HLRcompress
// Module      : compress
// Description : compression example
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <cstdio>
#include <cmath>

#include <hlrcompress/config.h>

#if USE_TBB == 1
#  include <tbb/parallel_for.h>
#  include <tbb/blocked_range2d.h>
#endif

#include <hlrcompress/compress.hh>
#include <hlrcompress/approx/svd.hh>
#include <hlrcompress/approx/rrqr.hh>

using namespace hlrcompress;

template < typename value_t >
blas::matrix< value_t >
gen_matrix_log ( const size_t  n )
{
    constexpr double  pi = 3.14159265358979323846;
    const     double  h  = 2 * pi / value_t(n);
    auto              M  = blas::matrix< value_t >( n, n );

    #if USE_TBB == 1
    
    ::tbb::parallel_for(
        ::tbb::blocked_range2d< size_t >( 0, n, 1024, 0, n, 1024 ),
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
    const auto  n = ( argc > 1 ? atoi( argv[1] ) : 128 );
    auto        M = gen_matrix_log< double >( n );

    auto  acc = fixed_prec( 1e-4 );
    auto  apx = SVD();
    auto  zM  = compress< double, decltype(apx) >( M, acc, apx, 32 );

    const auto  size_M  = M.byte_size();
    const auto  size_zM = zM->byte_size();
        
    std::cout << "original:   " << size_M << std::endl;
    std::cout << "compressed: " << size_zM << " / " << ( 100.0 * double(size_zM) / double(size_M) ) << "%" << std::endl;

    return 0;
}
