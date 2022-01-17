#ifndef __HLRCOMPRESS_DENSE_BLOCK_HH
#define __HLRCOMPRESS_DENSE_BLOCK_HH
//
// Project     : HLRcompress
// Module      : hlr/dense_block
// Description : represents a dense block
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hlrcompress/hlr/block.hh>
#include <hlrcompress/blas/arith.hh>
#include <hlrcompress/misc/compression.hh>
#include <hlrcompress/misc/error.hh>

namespace hlrcompress
{ 

template < typename T_value >
class dense_block : public block< T_value >
{
public:
    using value_t = T_value;
    using real_t  = real_type_t< value_t >;
    
    #if defined(HAS_ZFP)
    // compressed storage based on underlying floating point type
    using compressed_storage = std::unique_ptr< zfp::const_array2< real_t > >;
    #endif

private:
    // dense data
    blas::matrix< value_t >  _M;

    #if defined(HAS_ZFP)
    // optional: stores compressed data
    compressed_storage       _zM;
    #endif
    
public:
    //
    // ctors
    //

    dense_block ()
            : block< value_t >()
    {}
    
    dense_block ( const indexset  arow_is,
                  const indexset  acol_is )
            : block< value_t >( arow_is, acol_is )
    {}

    dense_block ( const indexset             arow_is,
                  const indexset             acol_is,
                  blas::matrix< value_t > &  aM )
            : block< value_t >( arow_is, acol_is )
            , _M( blas::copy( aM ) )
    {
        HLRCOMPRESS_ASSERT(( this->row_is().size() == _M.nrows() ) &&
                           ( this->col_is().size() == _M.ncols() ));
    }

    template < typename value_t >
    dense_block ( const indexset              arow_is,
                  const indexset              acol_is,
                  blas::matrix< value_t > &&  aM )
            : block< value_t >( arow_is, acol_is )
            , _M( std::move( aM ) )
    {
        HLRCOMPRESS_ASSERT(( this->row_is().size() == _M.nrows() ) &&
                           ( this->col_is().size() == _M.ncols() ));
    }

    // dtor
    virtual ~dense_block ()
    {}
    
    //
    // access internal data
    //

    blas::matrix< value_t > &        M ()       { return _M; }
    const blas::matrix< value_t > &  M () const { return _M; }
    
    void
    set_matrix ( const blas::matrix< value_t > &  aM )
    {
        HLRCOMPRESS_ASSERT(( this->nrows() == aM.nrows() ) &&
                           ( this->ncols() == aM.ncols() ));

        if ( is_compressed() )
            remove_compressed();
        
        blas::copy( aM, _M );
    }
    
    template < typename value_t >
    void
    set_matrix ( blas::matrix< value_t > &&  aM )
    {
        HLRCOMPRESS_ASSERT(( this->nrows() == aM.nrows() ) &&
                           ( this->ncols() == aM.ncols() ));

        if ( is_compressed() )
            remove_compressed();
        
        _M = std::move( aM );
    }

    //
    // structural information
    //
    
    // return true, if matrix is zero
    virtual bool    is_zero   () const { return false; } // full test too expensive
    
    //
    // compression functions
    //

    // compress internal data
    // - may result in non-compression if storage does not decrease
    virtual void   compress      ( const zconfig_t &  config )
    {
        #if defined(HAS_ZFP)
    
        if ( is_compressed() )
            return;
    
        uint          factor    = sizeof(value_t) / sizeof(real_t);
        const size_t  mem_dense = sizeof(value_t) * nrows() * ncols();
            
        if constexpr( std::is_same_v< value_t, real_t > )
        {
            auto  zM = std::make_unique< zfp::const_array2< value_t > >( _M.nrows(), _M.ncols(), config );
                
            zM->set( _M.data() );

            const size_t  mem_zfp = zM->compressed_size();

            if ( mem_zfp < mem_dense )
            {
                _zM = std::move( zM );
                _M  = std::move( blas::matrix< value_t >( 0, 0 ) );
            }// if
        }// if
        else
        {
            auto  zM = std::make_unique< zfp::const_array2< real_t > >( _M.nrows() * factor, _M.ncols(), config );
            
            zM->set( (real_t*) M.data() );
                
            const size_t  mem_zfp = zM->compressed_size();
                
            if ( mem_zfp < mem_dense )
            {
                _zM = std::move( zM );
                _M  = std::move( blas::matrix< value_t >( 0, 0 ) );
            }// if
        }// else

        #endif
    }

    // uncompress internal data
    virtual void   uncompress    ()
    {
        #if defined(HAS_ZFP)
        
        if ( ! is_compressed() )
            return;

        auto  uM = blas::matrix< value_t >( nrows(), ncols() );
    
        _zM->get( (real_t*) uM.data() );
        remove_compressed();
        
        _M = std::move( uM );
        
        #endif
    }

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        #if defined(HAS_ZFP)
        return _zM.get() != nullptr;
        #else
        return false;
        #endif
    }
    
    //
    // structural data
    //

    bool  is_structured () const { return false; }
    bool  is_lowrank    () const { return false; }
    bool  is_dense      () const { return true;  }

    //
    // misc.
    //
    
    // return size in bytes used by this object
    virtual size_t byte_size     () const
    {
        auto  bs = block< value_t >::byte_size() + _M.byte_size();

        #if defined(HAS_ZFP)

        bs += sizeof(_zM) + _zM.size();
        
        #endif

        return bs;
    }

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void   remove_compressed ()
    {
        #if defined(HAS_ZFP)
        _zM.reset( nullptr );
        #endif
    }
    
};

} // namespace hlrcompress

#endif // __HLRCOMPRESS_DENSE_BLOCK_HH
