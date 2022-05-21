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
#include <unistd.h>

#include <hdf5.h>

#include <hlrcompress/compress.hh>
#include <hlrcompress/approx/svd.hh>
#include <hlrcompress/approx/rrqr.hh>
#include <hlrcompress/approx/randsvd.hh>
#include <hlrcompress/hlr/error.hh>

using namespace hlrcompress;

using  value_t  = double;
using  my_clock = std::chrono::system_clock;

const blas::matrix< value_t >
read_matrix ( const std::string &  filename );

int
main ( int      argc,
       char **  argv )
{
    auto  matrix = std::string( "data.h5" );
    auto  acc    = double(1e-4);
    auto  ntile  = size_t(32);
    auto  zconf  = std::unique_ptr< zconfig_t >();
    auto  apx    = std::string( "svd" );
    uint  nbench = 1;
    char  opt;

    while (( opt = getopt( argc, argv, "i:e:t:a:p:r:l:hb:" )) != -1 )
    {
        switch (opt)
        {
            case 'i':
                matrix = optarg;
                break;
                
            case 'e':
                acc = atof( optarg );
                break;
                
            case 't':
                ntile = atoi( optarg );
                break;
                
            case 'l':
                apx = optarg;
                break;
                
            case 'a':
                zconf = std::make_unique< zconfig_t >( adaptive( atof( optarg ) ) );
                break;
                
            case 'p':
                zconf = std::make_unique< zconfig_t >( fixed_accuracy( atof( optarg ) ) );
                break;
                
            case 'r':
                zconf = std::make_unique< zconfig_t >( fixed_rate( atoi( optarg ) ) );
                break;

            case 'h' :
                std::cout << "h5compress -i data.h5 [options]" << std::endl
                          << std::endl
                          << "  -e eps  : relative accuracy of HLRcompress" << std::endl
                          << "  -t size : tile size of block layout" << std::endl
                          << "  -l apx  : low-rank approximation scheme (svd,rrqr,randsvd)" << std::endl
                          << "  -a fac  : adaptive ZFP compression" << std::endl
                          << "  -p fac  : fixed accuracy ZFP compression" << std::endl
                          << "  -r rate : fixed rate ZFP compression" << std::endl
                          << "  -b <n>  : run compress <n> times" << std::endl
                          << "  -h      : show this help" << std::endl;
                exit( 0 );
                
            case 'b' :
                nbench = atoi( optarg );
                break;
 
            default:
                std::cout << "unknown option, try -h" << std::endl;
                exit( 1 );
        }// switch
    }// while
    
    auto  M = read_matrix( matrix );
    
    std::cout << "compressing " << std::endl
              << "  matrix:     " << matrix << " ( " << M.nrows() << " x " << M.ncols() << " )" << std::endl
              << "  lowrank:    " << apx << std::endl
              << "  accuracy:   " << std::setprecision(4) << std::scientific << acc << std::endl
              << "  tilesize:   " << ntile << std::endl;

    if ( zconf.get() != nullptr )
        std::cout << "  zfp:        " << *zconf << std::endl;
    else
        std::cout << "  zfp:        none" << std::endl;

    auto    zM    = std::unique_ptr< block< value_t > >();
    double  t_min = -1;

    for ( uint  i = 0; i < nbench; ++i )
    {
        auto  tic = my_clock::now();

        if      ( apx == "svd"     ) zM = compress< value_t >( M, acc, SVD(), ntile, *zconf );
        else if ( apx == "rrqr"    ) zM = compress< value_t >( M, acc, RRQR(), ntile, *zconf );
        else if ( apx == "randsvd" ) zM = compress< value_t >( M, acc, RandSVD(), ntile, *zconf );
        else
            std::cout << "unknown low-rank approximation type : " << apx << std::endl;
        
        auto  toc = std::chrono::duration_cast< std::chrono::microseconds >( my_clock::now() - tic ).count() / 1e6;

        std::cout << "  runtime:    " << std::defaultfloat << toc << " s" << std::endl;
        
        if ( t_min < 0 ) t_min = toc;
        else             t_min = std::min( t_min, toc );

        if ( i != nbench - 1 )
            zM.reset( nullptr );
    }// for
    
    std::cout << "mintime:      " << std::defaultfloat << t_min << " s" << std::endl;

    const auto  bs_M  = M.byte_size();
    const auto  bs_zM = zM->byte_size();

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
read_matrix ( const std::string &  filename )
{
    auto  file      = H5Fopen( filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
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
