//
// Project     : HLRcompress
// Module      : compress
// Description : compression example
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

#include <hdf5.h>

#include <hlrcompress/compress.hh>
#include <hlrcompress/approx/svd.hh>
#include <hlrcompress/approx/rrqr.hh>
#include <hlrcompress/approx/randsvd.hh>
#include <hlrcompress/hlr/error.hh>

using namespace hlrcompress;

using value_t = double;

const blas::matrix< value_t >
read_matrix ( const char *  filename );

int
main ( int      argc,
       char **  argv )
{
    const auto  matrix = ( argc > 1 ? argv[1] : "data.h5" );
    auto        M      = read_matrix( matrix );

    const auto  acc    = ( argc > 2 ? atof( argv[2] ) : 1e-4 );
    const auto  ntile  = ( argc > 3 ? atoi( argv[3] ) : 32 );
    auto        apx    = SVD();

    // #if HLRCOMPRESS_USE_ZFP == 1
    
    // const auto  rate   = ( argc > 4 ? atoi( argv[4] ) : 0 );
    // auto        zconf  = ( rate > 0
    //                        ? std::make_unique< zconfig_t >( zfp_config_rate( rate, false ) )
    //                        : std::unique_ptr< zconfig_t >() );
    // #else
    
    // auto        zconf  = std::unique_ptr< zconfig_t >();
    
    // #endif
    
    std::cout << "compressing " << std::endl
              << "  matrix:     " << matrix << " ( " << M.nrows() << " x " << M.ncols() << " )" << std::endl
              << "  accuracy:   " << std::setprecision(4) << std::scientific << acc << std::endl
              << "  tilesize:   " << ntile << std::endl;

    // #if HLRCOMPRESS_USE_ZFP == 1
    
    // std::cout << "  zfp rate:   " << rate << std::endl;
    
    // #endif
    
    auto        tic    = std::chrono::high_resolution_clock::now();
    auto        zM     = compress< value_t, decltype(apx) >( M, acc, apx, ntile );
    auto        toc    = std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - tic );

    std::cout << "runtime:      " << std::defaultfloat << toc.count() / 1e6 << " s" << std::endl;

    const auto  bs_M   = M.byte_size();
    const auto  bs_zM  = zM->byte_size();

    std::cout << "storage " << std::endl
              << "  original:   " << bs_M << std::endl
              << "  compressed: " << bs_zM
              << " / " << ( 100.0 * double(bs_zM) / double(bs_M) ) << "%" 
              << " / " << double(bs_M) / double(bs_zM) << "x"
              << std::endl;

    // needs to be uncompressed for error computation (for now)
    zM->uncompress();

    auto  norm_M = blas::norm_F( M );

    if ( norm_M > 1e-32 )
        std::cout << "error:        " << std::setprecision(4) << std::scientific << error_fro( M, *zM ) / norm_M << std::endl;
    else
        std::cout << "error:        " << std::setprecision(4) << std::scientific << error_fro( M, *zM ) << std::endl;
    
    return 0;
}

//
// HDF5 read functions
//

herr_t
visit_func ( hid_t               /* loc_id */,
             const char *        name,
             const H5O_info_t *  info,
             void *              operator_data )
{
    std::string *  dname = static_cast< std::string * >( operator_data );
    std::string    oname = name;

    if ( oname[0] != '.' )
    {
        if ( info->type == H5O_TYPE_GROUP )
        {
            // use first name encountered
            if ( *dname == "" )
                *dname = oname;
        }// if
        else if ( info->type == H5O_TYPE_DATASET )
        {
            if ( *dname != "" )
            {
                if ( oname == *dname + "/value" )     // actual dataset
                    *dname = *dname + "/value";
                else if ( oname != *dname + "/type" ) // just type info
                    *dname = "";
            }// if
            else
            {
                *dname = oname;                       // directly use dataset
            }// else
        }// if
    }// if
    
    return 0;
}

const blas::matrix< value_t >
read_matrix ( const char *  filename )
{
    auto  file      = H5Fopen( filename, H5F_ACC_RDONLY, H5P_DEFAULT );
    auto  data_name = std::string( "" );
    auto  status    = H5Ovisit( file, H5_INDEX_NAME, H5_ITER_INC, visit_func, & data_name );

    if ( status != 0 )
        return blas::matrix< value_t >( 0, 0 );

    // check if any compatible type found
    if ( data_name == "" )
        return blas::matrix< value_t >( 0, 0 );

    // data_name = data_name + "/value";
    
    auto  dataset   = H5Dopen( file, data_name.c_str(), H5P_DEFAULT );
    auto  dataspace = H5Dget_space( dataset );
    auto  ndims     = H5Sget_simple_extent_ndims( dataspace );
    auto  dims      = std::vector< hsize_t >( ndims );

    if ( ndims != 2 )
        std::cout << "not a matrix" << std::endl;
                                                
    H5Sget_simple_extent_dims( dataspace, dims.data(), nullptr );
    
    // std::cout << dims[0] << " × " << dims[1] << std::endl;

    auto  M = blas::matrix< value_t >( dims[0], dims[1] );

    if constexpr ( std::is_same_v< value_t, double > )
        status = H5Dread( dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, M.data() );
    else 
        status = H5Dread( dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, M.data() );
    
    if ( status != 0 )
        return blas::matrix< value_t >( 0, 0 );
    
    return  M;
}
