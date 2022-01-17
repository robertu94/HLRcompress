#ifndef __HLRCOMPRESS_STRUCTURED_BLOCK_HH
#define __HLRCOMPRESS_STRUCTURED_BLOCK_HH
//
// Project     : HLRcompress
// Module      : structured_block
// Description : represents structured block with subblocks
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hlrcompress/hlr/block.hh>
#include <hlrcompress/misc/tensor.hh>
#include <hlrcompress/misc/error.hh>

namespace hlrcompress
{ 

template < typename T_value >
class structured_block : public block< T_value >
{
public:
    using value_t = T_value;
    using real_t  = real_type_t< value_t >;

private:
    // sub blocks
    tensor2< block< value_t > * >  _sub_blocks;
    
public:
    //
    // ctors
    //

    structured_block ()
            : block< value_t >()
    {}
    
    structured_block ( const indexset  arow_is,
                       const indexset  acol_is )
            : block< value_t >( arow_is, acol_is )
    {}

    // dtor
    virtual ~structured_block ()
    {
        for ( uint  i = 0; i < _sub_blocks.nrows(); ++i )
            for ( uint  j = 0; j < _sub_blocks.ncols(); ++j )
                delete _sub_blocks(i,j);
    }
    
    //
    // access internal data
    //

    block< value_t > &        sub_block ( const uint  i, const uint  j )       { return *_sub_blocks(i,j); }
    const block< value_t > &  sub_block ( const uint  i, const uint  j ) const { return *_sub_blocks(i,j); }
    
    void
    set_sub_block ( const uint          i,
                    const uint          j,
                    block< value_t > *  M )
    {
        _sub_blocks(i,j) = M;
    }
    
    //
    // structural information
    //
    
    // return true, if matrix is zero
    virtual bool    is_zero   () const { return false; }
    
    virtual size_t  nblock_rows () const { return _sub_blocks.nrows(); }
    virtual size_t  nblock_cols () const { return _sub_blocks.ncols(); }

    // set block structure
    void            set_block_struct ( const uint  nbrows,
                                       const uint  nbcols )
    {
        HLRCOMPRESS_ASSERT( nbrows * nbcols != 0 );

        //
        // copy old to new if possible and free otherwise
        //
        
        auto  new_subblocks = tensor2< block< value_t > * >( nbrows, nbcols );

        for ( uint  j = 0; j < std::min< uint >( _sub_blocks.ncols(), nbcols ); j++ )
            for ( uint  i = 0; i < std::min< uint >( _sub_blocks.nrows(), nbrows ); i++ )
                std::swap( new_subblocks(i,j), _sub_blocks(i,j) );

        _sub_blocks = std::move( new_subblocks );
    }
    
    //
    // compression functions
    //

    // compress internal data
    // - may result in non-compression if storage does not decrease
    virtual void   compress      ( const zconfig_t &  config )
    {
        for ( uint  i = 0; i < nblock_rows(); ++i )
            for ( uint  j = 0; j < nblock_cols(); ++j )
                sub_block( i, j ).compress( config );
    }

    // uncompress internal data
    virtual void   uncompress    ()
    {
        for ( uint  i = 0; i < nblock_rows(); ++i )
            for ( uint  j = 0; j < nblock_cols(); ++j )
                sub_block( i, j ).uncompress();
    }

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        for ( uint  i = 0; i < nblock_rows(); ++i )
            for ( uint  j = 0; j < nblock_cols(); ++j )
                if ( ! sub_block( i, j ).is_compressed() )
                    return false;

        return true;
    }
    
    //
    // structural data
    //

    bool  is_structured () const { return true;  }
    bool  is_lowrank    () const { return false; }
    bool  is_dense      () const { return false; }

    //
    // misc.
    //

    // return size in bytes used by this object
    virtual size_t byte_size  () const
    {
        size_t  bs = block< value_t >::byte_size() + sizeof(_sub_blocks) + sizeof(block< value_t >*) * nblock_rows() * nblock_cols();

        for ( uint  i = 0; i < nblock_rows(); ++i )
            for ( uint  j = 0; j < nblock_cols(); ++j )
                bs += sub_block(i,j).byte_size();

        return bs;
    }
};

} // namespace hlrcompress

#endif // __HLRCOMPRESS_STRUCTURED_BLOCK_HH
