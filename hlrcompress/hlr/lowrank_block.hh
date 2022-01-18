#ifndef __HLRCOMPRESS_LOWRANK_BLOCK_HH
#define __HLRCOMPRESS_LOWRANK_BLOCK_HH
//
// Project     : HLRcompress
// Module      : hlr/lowrank_block
// Description : represents low-rank block
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hlrcompress/hlr/block.hh>
#include <hlrcompress/blas/arith.hh>
#include <hlrcompress/misc/compression.hh>
#include <hlrcompress/misc/error.hh>

namespace hlrcompress
{ 

//
// Represents a low-rank matrix in factorised form: U·S·V^H
// with U and V represented as row/column cluster bases for
// corresponding matrix block (maybe joined by more matrices).
//
template < typename T_value >
class lowrank_block : public block< T_value >
{
public:
    using value_t = T_value;
    using real_t  = real_type_t< value_t >;
    
    #if USE_ZFP == 1
    // compressed storage based on underlying floating point type
    using compressed_storage = std::unique_ptr< zfp::const_array2< real_t > >;
    #endif

private:
    // lowrank factors
    blas::matrix< value_t >  _U, _V;

    #if USE_ZFP == 1
    // optional: stores compressed data
    compressed_storage       _zU, _zV;
    #endif

public:
    //
    // ctors
    //

    lowrank_block ()
            : block< value_t >()
    {}
    
    lowrank_block ( const indexset  arow_is,
                    const indexset  acol_is )
            : block< value_t >( arow_is, acol_is )
    {}

    template < typename value_t >
    lowrank_block ( const indexset             arow_is,
                    const indexset             acol_is,
                    blas::matrix< value_t > &  aU,
                    blas::matrix< value_t > &  aV )
            : block< value_t >( arow_is, acol_is )
            , _U( blas::copy( aU ) )
            , _V( blas::copy( aV ) )
    {
        HLRCOMPRESS_ASSERT(( this->row_is().size() == _U.nrows() ) &&
                           ( this->col_is().size() == _V.nrows() ) &&
                           ( _U.ncols()            == _V.ncols() ));
    }

    template < typename value_t >
    lowrank_block ( const indexset              arow_is,
                    const indexset              acol_is,
                    blas::matrix< value_t > &&  aU,
                    blas::matrix< value_t > &&  aV )
            : block< value_t >( arow_is, acol_is )
            , _U( std::move( aU ) )
            , _V( std::move( aV ) )
    {
        HLRCOMPRESS_ASSERT(( this->row_is().size() == _U.nrows() ) &&
                           ( this->col_is().size() == _V.nrows() ) &&
                           ( _U.ncols()            == _V.ncols() ));
    }

    // dtor
    virtual ~lowrank_block ()
    {}
    
    //
    // access internal data
    //

    size_t rank () const { return _U.ncols(); }

    blas::matrix< value_t > &        U ()       { return _U; }
    blas::matrix< value_t > &        V ()       { return _V; }
    
    const blas::matrix< value_t > &  U () const { return _U; }
    const blas::matrix< value_t > &  V () const { return _V; }
    
    void
    set_lrmat ( const blas::matrix< value_t > &  aU,
                const blas::matrix< value_t > &  aV )
    {
        HLRCOMPRESS_ASSERT(( this->nrows() == aU.nrows() ) &&
                           ( this->ncols() == aV.nrows() ) &&
                           ( aU.ncols()    == aV.ncols() ));

        if ( is_compressed() )
            remove_compressed();
        
        if ( aU.ncols() == _U.ncols() )
        {
            blas::copy( aU, _U );
            blas::copy( aV, _V );
        }// if
        else
        {
            _U = std::move( blas::copy( aU ) );
            _V = std::move( blas::copy( aV ) );
        }// else
    }
    
    void
    set_lrmat ( blas::matrix< value_t > &&  aU,
                blas::matrix< value_t > &&  aV )
    {
        HLRCOMPRESS_ASSERT(( this->nrows() == aU.nrows() ) &&
                           ( this->ncols() == aV.nrows() ) &&
                           ( aU.ncols()    == aV.ncols() ));

        if ( is_compressed() )
            remove_compressed();
        
        _U = std::move( aU );
        _V = std::move( aV );
    }

    //
    // structural information
    //
    
    // return true, if matrix is zero
    virtual bool    is_zero   () const { return ( rank() == 0 ); }
    
    //
    // compression functions
    //

    // compress internal data
    // - may result in non-compression if storage does not decrease
    virtual void   compress      ( const zconfig_t &  config )
    {
        #if USE_ZFP == 1
    
        if ( is_compressed() )
            return;
    
        uint          factor    = sizeof(value_t) / sizeof(real_t);
        const size_t  mem_dense = sizeof(value_t) * rank() * ( _U.nrows() + _V.ncols() );
            
        if constexpr( std::is_same_v< value_t, real_t > )
        {
            auto  zU = std::make_unique< zfp::const_array2< value_t > >( _U.nrows(), _U.ncols(), config );
            auto  zV = std::make_unique< zfp::const_array2< value_t > >( _V.nrows(), _V.ncols(), config );
                
            zU->set( _U.data() );
            zV->set( _V.data() );

            const size_t  mem_zfp = zU->compressed_size() + zV->compressed_size();

            if ( mem_zfp < mem_dense )
            {
                _zU = std::move( zU );
                _zV = std::move( zV );
                _U  = std::move( blas::matrix< value_t >( 0, rank() ) );
                _V  = std::move( blas::matrix< value_t >( 0, rank() ) );
            }// if
        }// if
        else
        {
            auto  zU = std::make_unique< zfp::const_array2< real_t > >( factor * _U.nrows(), _U.ncols(), config );
            auto  zV = std::make_unique< zfp::const_array2< real_t > >( factor * _V.nrows(), _V.ncols(), config );
                
            zU->set( (real_t *) _U.data() );
            zV->set( (real_t *) _V.data() );

            const size_t  mem_zfp = zU->compressed_size() + zV->compressed_size();

            if ( mem_zfp < mem_dense )
            {
                _zU = std::move( zU );
                _zV = std::move( zV );
                _U  = std::move( blas::matrix< value_t >( 0, rank() ) );
                _V  = std::move( blas::matrix< value_t >( 0, rank() ) );
            }// if
        }// else

        #endif
    }

    // uncompress internal data
    virtual void   uncompress    ()
    {
        #if USE_ZFP == 1
        
        if ( ! is_compressed() )
            return;

        auto  uU = blas::matrix< value_t >( this->nrows(), rank() );
        auto  uV = blas::matrix< value_t >( this->ncols(), rank() );
    
        _zU->get( (real_t*) uU.data() );
        _zV->get( (real_t*) uV.data() );
        remove_compressed();
        
        _U = std::move( uU );
        _V = std::move( uV );
        
        #endif
    }

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        #if USE_ZFP == 1
        return _zU.get() != nullptr;
        #else
        return false;
        #endif
    }

    //
    // structural data
    //

    bool  is_structured () const { return false; }
    bool  is_lowrank    () const { return true;  }
    bool  is_dense      () const { return false; }
    
    //
    // misc.
    //
    
    // return size in bytes used by this object
    virtual size_t byte_size  () const
    {
        auto  bs = block< value_t >::byte_size() + _U.byte_size() + _V.byte_size();

        #if USE_ZFP == 1

        bs += sizeof(_zU) + sizeof(_zV);

        if ( _zU.get() != nullptr )
            bs += _zU->size();
        
        if ( _zV.get() != nullptr )
            bs += _zV->size();
        
        #endif

        return bs;
    }

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void   remove_compressed ()
    {
        #if USE_ZFP == 1
        _zU.reset( nullptr );
        _zV.reset( nullptr );
        #endif
    }
};

} // namespace hlrcompress

#endif // __HLRCOMPRESS_LOWRANK_BLOCK_HH
