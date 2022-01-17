#ifndef __HLRCOMPRESS_BLOCK_HH
#define __HLRCOMPRESS_BLOCK_HH
//
// Project     : HLRcompress
// Module      : hlr/block
// Description : defines general block interface
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hlrcompress/hlr/indexset.hh>
#include <hlrcompress/blas/arith.hh>
#include <hlrcompress/misc/compression.hh>
#include <hlrcompress/misc/error.hh>

namespace hlrcompress
{ 

template < typename T_value >
class block
{
public:
    using value_t = T_value;
    using real_t  = real_type_t< value_t >;
    
private:
    // local index set of block
    indexset  _row_is, _col_is;

public:
    //
    // ctors
    //

    block ()
            : _row_is( 0, 0 )
            , _col_is( 0, 0 )
    {}
    
    block ( const indexset  arow_is,
            const indexset  acol_is )
            : _row_is( arow_is )
            , _col_is( acol_is )
    {}

    virtual ~block ()
    {}
    
    //
    // block structure 
    //
    
    size_t          nrows     () const { return _row_is.size(); }
    size_t          ncols     () const { return _col_is.size(); }

    indexset        row_is    () const { return _row_is; }
    indexset        col_is    () const { return _col_is; }
    
    // return true, if matrix is zero
    virtual bool    is_zero   () const = 0;
    
    //
    // compression functions
    //

    // compress internal data
    // - may result in non-compression if storage does not decrease
    virtual void   compress      ( const zconfig_t &  config ) = 0;

    // uncompress internal data
    virtual void   uncompress    () = 0;

    // return true if data is compressed
    virtual bool   is_compressed () const = 0;

    //
    // structural data
    //

    virtual bool  is_structured () const { return false; }
    virtual bool  is_lowrank    () const { return false; }
    virtual bool  is_dense      () const { return false; }
    
    //
    // misc.
    //
    
    // return size in bytes used by this object
    virtual size_t byte_size  () const
    {
        return _row_is.byte_size() + _col_is.byte_size();
    }
};

} // namespace hlrcompress

#endif // __HLRCOMPRESS_BLOCK_HH
