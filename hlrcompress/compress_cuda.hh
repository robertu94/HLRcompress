#ifndef __HLRCOMPRESS_COMPRESS_CUDA_HH
#define __HLRCOMPRESS_COMPRESS_CUDA_HH
//
// Project     : HLRcompress
// Module      : compress_cuda
// Description : level-wise compression function using CUDA
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hlrcompress/config.h>

#if HLRCOMPRESS_USE_CUDA == 1

#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#include <hlrcompress/blas/matrix.hh>
#include <hlrcompress/hlr/lowrank_block.hh>
#include <hlrcompress/hlr/dense_block.hh>
#include <hlrcompress/hlr/structured_block.hh>
#include <hlrcompress/misc/tensor.hh>
#include <hlrcompress/misc/error.hh>

// wrapper for cuda, cuBlas and cuSolver functions
#define HLRCOMPRESS_CUDA_CHECK( func, args )            \
    {                                                   \
        auto  result = func args ;                      \
        HLRCOMPRESS_ASSERT( result == cudaSuccess );    \
    }

#define HLRCOMPRESS_CUBLAS_CHECK( func, args )                  \
    {                                                           \
        auto  result = func args ;                              \
        HLRCOMPRESS_ASSERT( result == CUBLAS_STATUS_SUCCESS );  \
    }

#define HLRCOMPRESS_CUSOLVER_CHECK( func, args )                        \
    {                                                                   \
        auto  result = func args ;                                      \
        if ( result != CUSOLVER_STATUS_SUCCESS )                        \
            HLRCOMPRESS_ERROR( "cusolver result = " + std::to_string( int(result) ) ); \
    }


namespace hlrcompress
{

//
// uses batched SVD from cuSolver to approximate leaf level blocks
// and then proceed merging level by level upwards
//
// ATTENTION: proof of concept, not yet production ready
//
template < typename approx_t >
std::unique_ptr< block< double > >
compress_cuda ( blas::matrix< double > &  D,
                const double              rel_prec,
                const approx_t &          approx,
                const size_t              ntile,
                const zconfig_t *         zconf = nullptr )
{
    HLRCOMPRESS_ASSERT( D.nrows() == D.ncols() );
    
    const auto  norm_D = blas::norm_F( D );
    const auto  delta  = norm_D * rel_prec / D.nrows();
    auto        acc    = adaptive_accuracy( delta );
    
    //
    // first level, do batched SVD of all tiles
    //

    const size_t  nrows     = D.nrows();
    size_t        tilesize  = ntile;
    size_t        ntiles    = nrows / tilesize;

    //
    // batch limited number of block columns instead of full matrix
    //

    const size_t  nbcols    = std::min< size_t >( 16, ntiles );
    const size_t  batchsize = nbcols * ntiles;
    const size_t  fullsize  = ntiles * ntiles;

    std::vector< double >  A( tilesize*tilesize*batchsize );
    std::vector< double >  U( tilesize*tilesize*fullsize );
    std::vector< double >  V( tilesize*tilesize*fullsize );
    std::vector< double >  S( tilesize*fullsize );

    //
    // pre-allocate device memory
    //
    
    cusolverDnHandle_t       cusolverH     = nullptr;
    cudaStream_t             stream        = nullptr;
    gesvdjInfo_t             gesvdj_params = nullptr;

    const double             tol           = 1.e-7;
    const int                max_sweeps    = 15;
    const cusolverEigMode_t  jobz          = CUSOLVER_EIG_MODE_VECTOR;
    
    // create cusolver handle, bind a stream
    HLRCOMPRESS_CUSOLVER_CHECK( cusolverDnCreate, ( & cusolverH ) );
    HLRCOMPRESS_CUDA_CHECK( cudaStreamCreateWithFlags, ( &stream, cudaStreamNonBlocking ) );
    HLRCOMPRESS_CUSOLVER_CHECK( cusolverDnSetStream, ( cusolverH, stream ) );

    // configuration of gesvdj
    HLRCOMPRESS_CUSOLVER_CHECK( cusolverDnCreateGesvdjInfo, ( &gesvdj_params ) );
        
    // default value of tolerance is machine zero
    HLRCOMPRESS_CUSOLVER_CHECK( cusolverDnXgesvdjSetTolerance, ( gesvdj_params, tol ) );
        
    // default value of max. sweeps is 100
    HLRCOMPRESS_CUSOLVER_CHECK( cusolverDnXgesvdjSetMaxSweeps, ( gesvdj_params, max_sweeps ) );
        
    // step 3: copy A to device
    double *  d_A    = NULL; // lda-by-n-by-batchsize
    double *  d_U    = NULL; // ldu-by-m-by-batchsize
    double *  d_V    = NULL; // ldv-by-n-by-batchsize
    double *  d_S    = NULL; // minmn-by-batchsizee
    int *     d_info = NULL; // batchsize
    int       lwork  = 0;    // size of workspace
    double *  d_work = NULL; // device workspace for gesvdjBatched
        
    HLRCOMPRESS_CUDA_CHECK( cudaMalloc, ( (void**) & d_A   , sizeof(double)*tilesize*tilesize*batchsize ) );
    HLRCOMPRESS_CUDA_CHECK( cudaMalloc, ( (void**) & d_U   , sizeof(double)*tilesize*tilesize*batchsize ) );
    HLRCOMPRESS_CUDA_CHECK( cudaMalloc, ( (void**) & d_V   , sizeof(double)*tilesize*tilesize*batchsize ) );
    HLRCOMPRESS_CUDA_CHECK( cudaMalloc, ( (void**) & d_S   , sizeof(double)*tilesize*batchsize ) );
    HLRCOMPRESS_CUDA_CHECK( cudaMalloc, ( (void**) & d_info, sizeof(int   )*batchsize ) );
    
    // step 4: query working space of gesvdjBatched
    HLRCOMPRESS_CUSOLVER_CHECK( cusolverDnDgesvdjBatched_bufferSize,
                                ( cusolverH,
                                  jobz,
                                  tilesize, tilesize,
                                  d_A, tilesize,
                                  d_S,
                                  d_U, tilesize,
                                  d_V, tilesize,
                                  &lwork,
                                  gesvdj_params,
                                  batchsize ) );
    
    HLRCOMPRESS_CUDA_CHECK( cudaMalloc, ((void**)&d_work, sizeof(double)*lwork) );
    
    std::vector< int >  info( batchsize );
    
    //
    // loop over columns
    //

    size_t        bcol   = 0;
    const size_t  step_A = tilesize * tilesize;
    size_t        ofs_D  = 0;

    while ( bcol < ntiles )
    {
        for ( uint  bc = 0; bc < nbcols; ++bc )
        {
            for ( uint  j = 0; j < tilesize; ++j )
            {
                size_t  ofs_A = j * tilesize + ( bc * tilesize * tilesize * ntiles );
            
                for ( uint  i = 0; i < ntiles; ++i )
                {
                    memcpy( A.data() + ofs_A, D.data() + ofs_D, tilesize*sizeof(double) );
                    ofs_A += step_A;
                    ofs_D += tilesize;
                }// for
            }// for
        }// for

        //
        // copy to device
        //
        
        HLRCOMPRESS_CUDA_CHECK( cudaMemcpy, ( d_A, A.data(), sizeof(double)*A.size(), cudaMemcpyHostToDevice ) );
        HLRCOMPRESS_CUDA_CHECK( cudaDeviceSynchronize, () );
        
        // compute singular values of A0 and A1
        HLRCOMPRESS_CUSOLVER_CHECK( cusolverDnDgesvdjBatched,
                            ( cusolverH,
                              jobz,
                              tilesize, tilesize,
                              d_A, tilesize,
                              d_S,
                              d_U, tilesize,
                              d_V, tilesize,
                              d_work, lwork,
                              d_info,
                              gesvdj_params,
                              batchsize ) );
        HLRCOMPRESS_CUDA_CHECK( cudaDeviceSynchronize, () );

        //
        // copy back to host
        //

        const size_t  ofs_UV = tilesize*tilesize * bcol * ntiles;
        const size_t  ofs_S  = tilesize          * bcol * ntiles;

        HLRCOMPRESS_CUDA_CHECK( cudaMemcpy, ( U.data() + ofs_UV, d_U, sizeof(double)*tilesize*tilesize*batchsize, cudaMemcpyDeviceToHost) );
        HLRCOMPRESS_CUDA_CHECK( cudaMemcpy, ( V.data() + ofs_UV, d_V, sizeof(double)*tilesize*tilesize*batchsize, cudaMemcpyDeviceToHost) );
        HLRCOMPRESS_CUDA_CHECK( cudaMemcpy, ( S.data() + ofs_S,  d_S, sizeof(double)*tilesize*batchsize, cudaMemcpyDeviceToHost) );
        HLRCOMPRESS_CUDA_CHECK( cudaMemcpy, ( info.data(), d_info, sizeof(int) * batchsize, cudaMemcpyDeviceToHost) );
        
        //
        // proceed
        //

        bcol += nbcols;
    }// while
    
    //
    // construct lowrank/dense matrix blocks
    //
    
    auto  blocks   = tensor2< std::unique_ptr< block< double > > >( ntiles, ntiles );
    auto  acc_tile = acc( indexset( 0, tilesize-1 ), indexset( 0, tilesize-1 ) );
    auto  r        = ::tbb::blocked_range2d< size_t >( 0, ntiles, 0, ntiles );

    ::tbb::parallel_for( r,
        [&,tilesize] ( const auto &  r )
        {
            for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    auto  rowis = indexset( i * tilesize, (i+1) * tilesize - 1 );
                    auto  colis = indexset( j * tilesize, (j+1) * tilesize - 1 );

                    //
                    // determine approximation rank
                    //
            
                    const size_t  ofs_S = ( j * ntiles + i ) * tilesize;
                    auto          S_ij  = S.data() + ofs_S;
                    auto          sv    = blas::vector< double >( tilesize );

                    for ( size_t  l = 0; l < tilesize; ++l )
                        sv(l) = S_ij[l];
                    
                    auto  k = acc_tile.trunc_rank( sv );
                    
                    //
                    // decode if lowrank or dense and build matrix block for tile
                    //
                    
                    const size_t  ofs_A = ( j * ntiles + i ) * tilesize * tilesize;
                    
                    if ( k < tilesize/2 )
                    {
                        auto  U_ij = blas::matrix< double >( tilesize, k );
                        auto  V_ij = blas::matrix< double >( tilesize, k );
                        
                        for ( uint  l = 0; l < k; ++l )
                        {
                            for ( uint  ii = 0; ii < tilesize; ++ii )
                                U_ij( ii, l ) = sv(l) * U.data()[ofs_A + (l*tilesize) + ii];
                            for ( uint  ii = 0; ii < tilesize; ++ii )
                                V_ij( ii, l ) = V.data()[ofs_A + (l*tilesize) + ii];
                        }// for
                        
                        blocks( i, j ) = std::make_unique< lowrank_block< double > >( rowis, colis, std::move( U_ij ), std::move( V_ij ) );
                    }// if
                    else
                    {
                        auto  D_ij  = blas::copy( D( blas::range( i*tilesize, (i+1)*tilesize - 1 ),
                                                     blas::range( j*tilesize, (j+1)*tilesize - 1 ) ) );

                        blocks( i, j ) = std::make_unique< dense_block< double > >( rowis, colis, std::move( blas::copy( D_ij ) ) );
                    }// else
                }// for
            }// for
        } );
    
    //
    // iterate upwards till the roof
    //
    
    while ( true )
    {
        const auto  tilesize_up = tilesize * 2;
        const auto  ntiles_up   = ntiles / 2;

        //
        // join 2x2 small blocks into a larger block
        //
        //   mapping:  larger (i,j) -> small (2*i,  2*j) (2*i,   2*j+1)
        //                                   (2*i+1,2*j) (2*i+1, 2*j+1)
        //

        auto  blocks_up = tensor2< std::unique_ptr< block< double > > >( ntiles_up, ntiles_up );

        ::tbb::parallel_for(
            ::tbb::blocked_range2d< size_t >( 0, ntiles_up,
                                              0, ntiles_up ),
            [&,tilesize_up] ( const auto &  r )
            {
                for ( auto  ii = r.rows().begin(); ii != r.rows().end(); ++ii )
                {
                    for ( auto  jj = r.cols().begin(); jj != r.cols().end(); ++jj )
                    {
                        auto  rowis       = indexset( ii * tilesize_up, (ii+1) * tilesize_up - 1 );
                        auto  colis       = indexset( jj * tilesize_up, (jj+1) * tilesize_up - 1 );
                
                        auto  ofs_i       = 2*ii;
                        auto  ofs_j       = 2*jj;
                        auto  sub_blocks  = tensor2< std::unique_ptr< block< double > > >( 2, 2 );
                        bool  all_lowrank = true;
                        bool  all_dense   = true;

                        for ( uint  i = 0; i < 2; ++i )
                        {
                            for ( uint  j = 0; j < 2; ++j )
                            {
                                sub_blocks(i,j) = std::move( blocks( ofs_i + i, ofs_j + j ) );

                                HLRCOMPRESS_ASSERT( sub_blocks(i,j).get() != nullptr );

                                if ( ! sub_blocks(i,j)->is_lowrank() )
                                    all_lowrank = false;

                                if ( ! sub_blocks(i,j)->is_dense() )
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
                                    rank_sum += static_cast< lowrank_block< double > * >( sub_blocks(i,j).get() )->rank();

                            // copy sub block data into global structure
                            auto    U    = blas::matrix< double >( rowis.size(), rank_sum );
                            auto    V    = blas::matrix< double >( colis.size(), rank_sum );
                            auto    pos  = 0; // pointer to next free space in U/V
                            size_t  smem = 0; // holds memory of sub blocks
            
                            for ( uint  i = 0; i < 2; ++i )
                            {
                                for ( uint  j = 0; j < 2; ++j )
                                {
                                    auto  Rij   = static_cast< lowrank_block< double > * >( sub_blocks(i,j).get() );
                                    auto  Uij   = Rij->U();
                                    auto  Vij   = Rij->V();
                                    auto  U_sub = U( Rij->row_is() - rowis.first(), blas::range( pos, pos + Uij.ncols() - 1 ) );
                                    auto  V_sub = V( Rij->col_is() - colis.first(), blas::range( pos, pos + Uij.ncols() - 1 ) );

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
                                blocks_up( ii, jj ) = std::make_unique< lowrank_block< double > >( rowis, colis, std::move( W ), std::move( X ) );
                            }// if
                        }// if
                        else if ( all_dense )
                        {
                            //
                            // always join dense blocks
                            //
        
                            auto  D = blas::matrix< double >( rowis.size(), colis.size() );
                        
                            for ( uint  i = 0; i < 2; ++i )
                            {
                                for ( uint  j = 0; j < 2; ++j )
                                {
                                    auto  sub_ij = static_cast< dense_block< double > * >( sub_blocks(i,j).get() );
                                    auto  sub_D = D( sub_ij->row_is() - rowis.first(),
                                                     sub_ij->col_is() - colis.first() );

                                    blas::copy( sub_ij->M(), sub_D );
                                }// for
                            }// for
                    
                            blocks_up( ii, jj ) = std::make_unique< dense_block< double > >( rowis, colis, std::move( D ) );
                        }// if

                        if ( blocks_up( ii, jj ) == nullptr )
                        {
                            //
                            // either not all low-rank or memory gets larger: construct block matrix
                            //

                            auto  B = std::make_unique< structured_block< double > >( rowis, colis );

                            B->set_block_struct( 2, 2 );
        
                            for ( uint  i = 0; i < 2; ++i )
                            {
                                for ( uint  j = 0; j < 2; ++j )
                                {
                                    if (( zconf != nullptr ) && ! sub_blocks(i,j)->is_structured() )
                                        sub_blocks(i,j)->compress( *zconf );
                
                                    B->set_sub_block( i, j, sub_blocks(i,j).release() );
                                }// for
                            }// for

                            blocks_up( ii, jj ) = std::move( B );
                        }// if
                    }// for
                }// for
            } );

        blocks = std::move( blocks_up );

        if ( ntiles_up == 1 )
            break;

        tilesize = tilesize_up;
        ntiles   = ntiles_up;
    }// while
    
    //
    // return single, top-level matrix in "blocks"
    //
    
    return std::move( blocks( 0, 0 ) );
}

}// namespace hlrcompress

#endif // HLRCOMPRESS_USE_CUDA

#endif // __HLRCOMPRESS_COMPRESS_CUDA_HH
