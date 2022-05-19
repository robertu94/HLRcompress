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
    
    #if HLRCOMPRESS_USE_ZFP == 1
    // compressed storage based on underlying floating point type
    using compressed_storage = zarray;
    #endif

private:
    // dense data
    blas::matrix< value_t >  _M;

    #if HLRCOMPRESS_USE_ZFP == 1
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
        #if HLRCOMPRESS_USE_ZFP == 1
    
        if ( is_compressed() )
            return;

        using real_t = real_type_t< value_t >;
        
        constexpr auto  factor    = sizeof(value_t) / sizeof(real_t);
        const size_t    mem_dense = sizeof(value_t) * _M.nrows() * _M.ncols();

        if ( config.mode != compress_adaptive )
        {
            auto  zM = zcompress< real_t >( config, (real_t *) _M.data(), _M.nrows() * factor, _M.ncols() );

            if ( zM.size() < mem_dense )
            {
                _zM = std::move( zM );
                _M  = std::move( blas::matrix< value_t >( 0, 0 ) );
            }// if
        }// if
        else
        {
            const double  tol = config.accuracy; // * std::sqrt( double(_M.nrows()) * double(_M.ncols()) );
            auto          T   = blas::matrix< value_t >( this->nrows(), this->ncols() );
                                   
            // for ( uint  rate = 8; rate <= 64; rate += 2 )
            // {
            //     auto loc_cfg = fixed_rate( rate );
            for ( double  acc = tol * 1e1; acc >= tol * 1e-3; acc *= 0.5 )
            {
                auto loc_cfg = fixed_accuracy( acc );
                auto zM      = zcompress< real_t >( loc_cfg, (real_t *) _M.data(), _M.nrows() * factor, _M.ncols() );
                
                zuncompress< real_t >( zM, (real_t *) T.data(), T.nrows() * factor, T.ncols() );
                
                blas::add( value_t(-1), _M, T );
                
                const auto  error = blas::norm_F( T );
                
                if ( error <= tol )
                {
                    if ( zM.size() < mem_dense )
                    {
                        _zM = std::move( zM );
                        _M  = std::move( blas::matrix< value_t >( 0, 0 ) );
                    }// if
                    
                    return;
                }// if
            }// while
        }// else
        
        #endif
    }

    // uncompress internal data
    virtual void   uncompress    ()
    {
        #if HLRCOMPRESS_USE_ZFP == 1
        
        if ( ! is_compressed() )
            return;

        using real_t = real_type_t< value_t >;
        
        constexpr uint  factor = sizeof(value_t) / sizeof(real_t);
        auto            uM     = blas::matrix< value_t >( this->nrows(), this->ncols() );
    
        zuncompress< real_t >( _zM, (real_t *) uM.data(), uM.nrows() * factor, uM.ncols() );
        remove_compressed();
        
        _M = std::move( uM );
        
        #endif
    }

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        #if HLRCOMPRESS_USE_ZFP == 1
        return _zM.data() != nullptr;
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

        #if HLRCOMPRESS_USE_ZFP == 1

        bs += sizeof(_zM);

        if ( is_compressed() )
            bs +=_zM.size();
        
        #endif

        return bs;
    }

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void   remove_compressed ()
    {
        #if HLRCOMPRESS_USE_ZFP == 1
        _zM = std::move( zarray() );
        #endif
    }
    
};

} // namespace hlrcompress

#endif // __HLRCOMPRESS_DENSE_BLOCK_HH
