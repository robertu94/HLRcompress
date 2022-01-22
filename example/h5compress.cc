//
// Project     : HLRcompress
// Module      : compress
// Description : compression example
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <vector>
#include <string>

#include <H5Cpp.h>

#include <hlrcompress/compress.hh>
#include <hlrcompress/approx/svd.hh>
#include <hlrcompress/approx/rrqr.hh>

using namespace hlrcompress;

const blas::matrix< double >
read_matrix ( const char *  filename );

int
main ( int      argc,
       char **  argv )
{
    const auto  matrix = ( argc > 1 ? argv[1] : "matrix.h5" );
    auto        M      = read_matrix( matrix );

    std::cout << "matrix " << matrix << " has dimension " << M.nrows() << " x " << M.ncols() << std::endl;
    
    auto  acc = fixed_prec( 1e-4 );
    auto  apx = SVD();
    auto  zM  = compress< double, decltype(apx) >( M, acc, apx, 32 );

    const auto  size_M  = M.byte_size();
    const auto  size_zM = zM->byte_size();
        
    std::cout << "original:   " << size_M << std::endl;
    std::cout << "compressed: " << size_zM << " / " << ( 100.0 * double(size_zM) / double(size_M) ) << "%" << std::endl;

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

const blas::matrix< double >
read_matrix ( const char *  filename )
{
    auto  file      = H5Fopen( filename, H5F_ACC_RDONLY, H5P_DEFAULT );
    auto  data_name = std::string( "" );
    auto  status    = H5Ovisit( file, H5_INDEX_NAME, H5_ITER_INC, visit_func, & data_name );

    if ( status != 0 )
        return blas::matrix< double >( 0, 0 );

    // check if any compatible type found
    if ( data_name == "" )
        return blas::matrix< double >( 0, 0 );

    // data_name = data_name + "/value";
    
    auto  dataset   = H5Dopen( file, data_name.c_str(), H5P_DEFAULT );
    auto  dataspace = H5Dget_space( dataset );
    auto  ndims     = H5Sget_simple_extent_ndims( dataspace );
    auto  dims      = std::vector< hsize_t >( ndims );

    if ( ndims != 2 )
        std::cout << "not a matrix" << std::endl;
                                                
    H5Sget_simple_extent_dims( dataspace, dims.data(), nullptr );
    
    // std::cout << dims[0] << " × " << dims[1] << std::endl;

    auto  M = blas::matrix< double >( dims[0], dims[1] );

    status = H5Dread( dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, M.data() );
    
    if ( status != 0 )
        return blas::matrix< double >( 0, 0 );
    
    return  M;
}
