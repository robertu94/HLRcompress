#ifndef __HLRCOMPRESS_BLAS_VECTOR_HH
#define __HLRCOMPRESS_BLAS_VECTOR_HH
//
// Project     : HLRcompress
// Module      : blas/vector
// Description : implements dense vector class for BLAS operations
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include "hlrcompress/blas/vector_base.hh"
#include "hlrcompress/blas/memblock.hh"
#include "hlrcompress/misc/error.hh"

namespace hlrcompress { namespace blas {

template < typename value_t > class matrix;

//
// Standard vector class in linear algebra routines. It holds a
// reference to a memory block, which is controlled by this vector,
// e.g. if the vector is gone or resized, the memory block is also gone.
//
template < typename T_value >
class vector : public vector_base< vector< T_value > >, public memblock< T_value >
{
public:
    // internal value type
    using   value_t = T_value;

    // super class type
    using   super_t = memblock< value_t >;
    
private:
    // length of vector
    size_t   _length;

    // stride of data in memory block
    size_t   _stride;
    
public:
    //
    // constructor and destructor
    //

    // create zero sized vector
    vector () noexcept
            : memblock< value_t >()
            , _length( 0 )
            , _stride( 0 )
    {}

    // create vector of size \a n
    vector ( const size_t  n )
            : memblock<value_t>( n )
            , _length( n )
            , _stride( 1 )
    {}

    // copy ctor
    vector ( const vector &       v,
             const copy_policy_t  p = copy_reference )
            : memblock< value_t >()
            , _length( 0 )
            , _stride( 0 )
    {
        switch ( p )
        {
            case copy_reference :
                (*this) = v;
                break;

            case copy_value :
                super_t::alloc_wo_value( v.length() );
                _length = v.length();
                _stride = 1;
                
                for ( idx_t i = 0; i < idx_t( _length ); i++ )
                    (*this)(i) = v(i);
                
                break;
        }// switch
    }
    
    // move ctor
    vector ( vector &&  v ) noexcept
            : memblock< value_t >( std::move( v ) )
            , _length( 0 )
            , _stride( 0 )
    {
        v._data = nullptr;
        
        std::swap( _length, v._length );
        std::swap( _stride, v._stride );
    }

    // create copy of sub vector of \a v defined by range \a r
    vector ( const vector< value_t > & v,
             const range &             ar,
             const copy_policy_t       p = copy_reference )
            : memblock< value_t >()
            , _length( 0 )
            , _stride( 0 )
    {
        const range  r( ar == range::all ? range( 0, idx_t(v.length())-1 ) : ar );
        
        HLRCOMPRESS_DBG_ASSERT( r.size() <= v.length() );

        _length = r.size();

        switch ( p )
        {
            case copy_reference :
                _stride = v.stride();
                super_t::init( v.data() + r.first() * v.stride() );
                break;

            case copy_value :
                super_t::alloc_wo_value( _length );
                _stride = 1;
                for ( idx_t  i = 0; i < idx_t(_length); ++i )
                    (*this)(i) = v( r.first() + i );
                break;
        }// switch
    }

    // create copy of column of matrix \a M defined by \a r and \a col
    vector ( const matrix< value_t > &  M,
             const range &              ar,
             const idx_t                col,
             const copy_policy_t        p = copy_reference )
            : memblock< value_t >()
            , _length( 0 )
            , _stride( 0 )
    {
        const range  r( ar == range::all ? M.row_range() : ar );

        if ( r.size() > M.nrows() )
            HLRCOMPRESS_ERROR( "range size × range stride > matrix rows" );

        _length = r.size();

        switch ( p )
        {
            case copy_reference :
                _stride = M.row_stride();
                super_t::init( M.data() + r.first() * M.row_stride() + col * M.col_stride() );
                break;
                
            case copy_value :
                super_t::alloc_wo_value( _length );
                _stride = 1;
                for ( idx_t  i = 0; i < idx_t(_length); i++ )
                    (*this)(i) = M( r.first() + i, col );
                break;
        }// switch
    }
    
    // create copy/reference to row of matrix \a M defined by \a r and \a row
    vector ( const matrix< value_t > &  M,
             const idx_t                row,
             const range &              ar,
             const copy_policy_t        p = copy_reference )
            : memblock< value_t >()
            , _length( 0 )
            , _stride( 0 )
    {
        const range  r( ar == range::all ? M.col_range() : ar );
        
        if ( r.size() > M.ncols() )
            HLRCOMPRESS_ERROR( "range size × range stride > matrix columns" );

        _length = r.size();

        switch ( p )
        {
            case copy_reference :
                _stride = M.col_stride();
                super_t::init( M.data() + r.first() * M.col_stride() + row * M.row_stride() );
                break;
                
            case copy_value :
                super_t::alloc_wo_value( _length );
                _stride = 1;
        
                for ( idx_t  i = 0; i < idx_t(_length); i++ )
                    (*this)(i) = M( row, r.first() + i );
                break;
        }// switch
    }

    // copy operator (always copy reference! for real data copy, use copy function below!)
    vector & operator = ( const vector & v )
    {
        super_t::init( v.data(), false );
        _length = v.length();
        _stride = v.stride();
        
        return *this;
    }

    // move operator (move ownership of data)
    vector & operator = ( vector && v ) noexcept
    {
        if ( this != & v ) // prohibit self-moving
        {
            super_t::init( v, v._is_owner );
            _length = v.length();
            _stride = v.stride();

            // reset data of v
            v._data   = nullptr;
            v._length = 0;
            v._stride = 0;
        }// if
        
        return *this;
    }
    
    //
    // data access
    //

    // return length of vector
    size_t    length      () const noexcept { return _length; }

    // return stride of index set
    size_t    stride      () const noexcept { return _stride; }
    
    // return coefficient at position \a i
    value_t   operator () ( const idx_t  i ) const noexcept
    {
        HLRCOMPRESS_DBG_ASSERT( i < idx_t(_length) );
        return super_t::_data[ i * _stride ];
    }
    
    // return reference to coefficient at position \a i
    value_t & operator () ( const idx_t  i ) noexcept
    {
        HLRCOMPRESS_DBG_ASSERT( i < idx_t(_length) );
        return super_t::_data[ i * _stride ];
    }

    //
    // construction operators
    //

    // create real copy of matrix
    vector< value_t >  copy () const
    {
        vector< value_t >  v( *this, copy_value );

        return v;
    }
    
    // create reference to this matrix
    vector< value_t >  reference () const
    {
        vector< value_t >  v( *this, copy_reference );

        return v;
    }
    
    // return reference to sub vector defined by \a r
    vector    operator () ( const range & r ) const
    {
        return vector( *this, r, copy_reference );
    }

    // give access to internal data
    value_t * data     () const noexcept { return super_t::_data; }

    // return size in bytes used by this object
    size_t  byte_size () const
    {
        return sizeof( value_t ) * _length + sizeof(_length) + sizeof(_stride);
    }
};

//
// return real copy of given vector
//
template < typename value_t >
vector< value_t >
copy ( const vector< value_t > &  v )
{
    return v.copy();
}

//
// gives access to vector value type
//
template < typename T_value >
struct vector_value_type< vector< T_value > >
{
    using  value_t = T_value;
};

//
// signals vector type
//
template < typename value_t >
struct is_vector< vector< value_t > >
{
    static const bool  value = true;
};

//
// create random vector
//
template < typename T >
vector< T >
random ( const size_t  length );

}}// namespace hlrcompress::blas

#endif  // __HLRCOMPRESS_BLAS_VECTOR_HH
