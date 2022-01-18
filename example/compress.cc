//
// Project     : HLRcompress
// Module      : compress
// Description : compression example
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <random>

#include <hlrcompress/compress.hh>
#include <hlrcompress/approx/svd.hh>
#include <hlrcompress/approx/rrqr.hh>

int
main ( int,
       char ** )
{
    const auto  n             = 100;
    auto        M             = hlrcompress::blas::matrix< double >( n, n );
    auto        rd            = std::random_device{};
    auto        generator     = std::mt19937{ rd() };
    auto        uniform_distr = std::uniform_real_distribution<>{ 0, 1 };

    for ( size_t  j = 0; j < n; ++j )
        for ( size_t  i = 0; i < n; ++i )
            M(i,j) = uniform_distr( generator );

    auto  acc = hlrcompress::fixed_prec( 1e-4 );
    auto  apx = hlrcompress::RRQR();
    auto  zM  = hlrcompress::compress< double, decltype(apx) >( M, acc, apx, 16 );

    std::cout << M.byte_size() << std::endl;
    std::cout << zM->byte_size() << std::endl;

    return 0;
}
