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
    
    #if HLRCOMPRESS_USE_ZFP == 1
    // compressed storage based on underlying floating point type
    using compressed_storage = zarray;
    #endif

private:
    // lowrank factors
    blas::matrix< value_t >  _U, _V;

    #if HLRCOMPRESS_USE_ZFP == 1
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
        #if HLRCOMPRESS_USE_ZFP == 1
    
        if ( is_compressed() )
            return;
    
        using real_t = real_type_t< value_t >;

        constexpr auto  factor    = sizeof(value_t) / sizeof(real_t);
        const size_t    mem_dense = sizeof(value_t) * rank() * ( _U.nrows() + _V.ncols() );
        auto            zU        = zcompress< real_t >( config, (real_t *) _U.data(), _U.nrows() * factor, _U.ncols() );
        auto            zV        = zcompress< real_t >( config, (real_t *) _V.data(), _V.nrows() * factor, _V.ncols() );
            
        if ( zU.size() + zV.size() < mem_dense )
        {
            _zU = std::move( zU );
            _zV = std::move( zV );
            _U  = std::move( blas::matrix< value_t >( 0, rank() ) );
            _V  = std::move( blas::matrix< value_t >( 0, rank() ) );
        }// if

        #endif
    }

    // uncompress internal data
    virtual void   uncompress    ()
    {
        #if HLRCOMPRESS_USE_ZFP == 1
        
        if ( ! is_compressed() )
            return;

        using real_t = real_type_t< value_t >;
        
        constexpr auto  factor = sizeof(value_t) / sizeof(real_t);
        auto            uU     = blas::matrix< value_t >( this->nrows(), rank() );
        auto            uV     = blas::matrix< value_t >( this->ncols(), rank() );
    
        zuncompress< real_t >( _zU, (real_t *) uU.data(), uU.nrows() * factor, uU.ncols() );
        zuncompress< real_t >( _zV, (real_t *) uV.data(), uV.nrows() * factor, uV.ncols() );
        remove_compressed();
        
        _U = std::move( uU );
        _V = std::move( uV );
        
        #endif
    }

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        #if HLRCOMPRESS_USE_ZFP == 1
        return _zU.data() != nullptr;
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

        #if HLRCOMPRESS_USE_ZFP == 1

        bs += sizeof(_zU) + sizeof(_zV);

        if ( is_compressed() )
        {
            bs += _zU.size();
            bs += _zV.size();
        }// if
        
        #endif

        return bs;
    }

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void   remove_compressed ()
    {
        #if HLRCOMPRESS_USE_ZFP == 1
        _zU = std::move( zarray() );
        _zV = std::move( zarray() );
        #endif
    }
};

} // namespace hlrcompress

#endif // __HLRCOMPRESS_LOWRANK_BLOCK_HH
