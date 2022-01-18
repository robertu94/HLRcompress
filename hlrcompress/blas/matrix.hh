#ifndef __HLRCOMPRESS_BLAS_MATRIX_HH
#define __HLRCOMPRESS_BLAS_MATRIX_HH
//
// Project     : HLRcompress
// Module      : blas/matrix
// Description : implements dense matrix class for BLAS/LAPACK operations
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hlrcompress/blas/memblock.hh>
#include <hlrcompress/blas/matrix_base.hh>

namespace hlrcompress { namespace blas {

//
// dense matrix for interfacing with blas/LAPACK 
//
template < typename T_value >
class matrix : public matrix_base< matrix< T_value > >, public memblock< T_value >
{
public:
    // internal value type
    using  value_t = T_value;

    // super class type
    using  super_t = memblock< value_t >;
    
private:
    // dimensions of matrix
    size_t   _length[2];

    // strides of data in memory block (rows and columns)
    size_t   _stride[2];

public:
    //
    // constructor and destructor
    //

    // creates zero sized matrix
    matrix () noexcept
            : memblock< value_t >()
            , _length{ 0, 0 }
            , _stride{ 0, 0 }
    {}

    // creates matrix of size \a anrows × \a ancols
    matrix ( const size_t  anrows,
             const size_t  ancols )
            : memblock< value_t >( anrows * ancols )
            , _length{ anrows, ancols }
            , _stride{ 1, anrows }
    {}

    // copy constructor
    matrix ( const matrix &       M,
             const copy_policy_t  p = copy_reference )
            : memblock< value_t >()
            , _length{ 0, 0 }
            , _stride{ 0, 0 }
    {
        switch ( p )
        {
            case copy_reference :
                (*this) = M;
                break;

            case copy_value :
                _length[0] = M._length[0];
                _length[1] = M._length[1];
                _stride[0] = 1;
                _stride[1] = _length[0];
                super_t::alloc_wo_value( _length[0] * _length[1] );

                for ( idx_t j = 0; j < idx_t( _length[1] ); j++ )
                    for ( idx_t i = 0; i < idx_t( _length[0] ); i++ )
                        (*this)(i,j) = M( i, j );
                
                break;
        }// switch
    }

    // copy constructor for other matrix types
    template < typename matrix_t >
    matrix ( const matrix_t &  M )
            : memblock< value_t >()
            , _length{ 0, 0 }
            , _stride{ 0, 0 }
    {
        (*this) = M;
    }
    
    // move constructor
    matrix ( matrix &&  M ) noexcept
            : memblock< value_t >( std::move( M ) )
            , _length{ 0, 0 }
            , _stride{ 0, 0 }
    {
        M._data = nullptr;
        
        std::swap( _length, M._length );
        std::swap( _stride, M._stride );
    }

    // creates matrix using part of \a M defined by \a r1 × \a r2
    // \a p defines whether data is copied or referenced
    matrix ( const matrix &       M,
             const range &        ar1,
             const range &        ar2,
             const copy_policy_t  p = copy_reference )
            : memblock< value_t >()
            , _length{ 0, 0 }
            , _stride{ 0, 0 }
    {
        const range  r1( ar1 == range::all ? M.row_range() : ar1 );
        const range  r2( ar2 == range::all ? M.col_range() : ar2 );
        
        _length[0] = r1.size();
        _length[1] = r2.size();

        switch ( p )
        {
            case copy_reference :
                _stride[0] = M.row_stride();
                _stride[1] = M.col_stride();
            
                super_t::init( M.data() + r1.first() * M.row_stride() + r2.first() * M.col_stride() );
                break;

            case copy_value :
                super_t::alloc_wo_value( _length[0] * _length[1] );
                _stride[0] = 1;
                _stride[1] = _length[0];

                for ( idx_t j = 0; j < idx_t( _length[1] ); j++ )
                    for ( idx_t i = 0; i < idx_t( _length[0] ); i++ )
                        (*this)(i,j) = M( r1.first() + i, r2.first() + j );
                break;
        }// switch
    }

    // special version for matrix views as above version leads to infinite loop
    // only copy_value is supported!
    template < typename matrix_t >
    matrix ( const matrix_base< matrix_t > &  M,
             const range &        ar1,
             const range &        ar2,
             const copy_policy_t  p = copy_reference )
            : memblock< value_t >()
            , _length{ 0, 0 }
            , _stride{ 0, 0 }
    {
        const range  r1( ar1 == range::all ? M.row_range() : ar1 );
        const range  r2( ar2 == range::all ? M.col_range() : ar2 );
        
        _length[0] = r1.size();
        _length[1] = r2.size();

        switch ( p )
        {
            case copy_reference :
                HLRCOMPRESS_ERROR( "copy_reference not supported" );
                break;

            case copy_value :
                super_t::alloc_wo_value( _length[0] * _length[1] );
                _stride[0] = 1;
                _stride[1] = _length[0];

                for ( idx_t j = 0; j < idx_t( _length[1] ); j++ )
                    for ( idx_t i = 0; i < idx_t( _length[0] ); i++ )
                        (*this)(i,j) = M( r1.first() + i, r2.first() + j );
                break;
        }// switch
    }

    // copy operator for matrices (always copy reference! for real copy, use blas::copy)
    matrix & operator = ( const matrix &  M )
    {
        super_t::init( M.data(), false );
        _length[0] = M.nrows();
        _length[1] = M.ncols();
        _stride[0] = M.row_stride();
        _stride[1] = M.col_stride();

        return *this;
    }

    // move operator
    matrix & operator = ( matrix &&  M ) noexcept
    {
        if ( this != & M ) // prohibit self-moving
        {
            super_t::init( M, M._is_owner );
            
            _length[0] = M.nrows();
            _length[1] = M.ncols();
            _stride[0] = M.row_stride();
            _stride[1] = M.col_stride();

            M._data   = nullptr;
            M._length[0] = 0;
            M._length[1] = 0;
            M._stride[0] = 0;
            M._stride[1] = 0;
        }// if

        return *this;
    }
    
    //
    // data access
    //

    // return number of rows of matrix
    size_t       nrows        () const noexcept { return _length[0]; }
    
    // return number of columns of matrix
    size_t       ncols        () const noexcept { return _length[1]; }

    // return blas matrix view of matrix object
    blasview_t   blas_view    () const noexcept { return BLAS_NORMAL; }

    // return number of rows of actual blas matrix
    size_t       blas_nrows   () const noexcept { return nrows(); }

    // return number of columns of actual blas matrix
    size_t       blas_ncols   () const noexcept { return ncols(); }

    // return coefficient (i,j)
    value_t      operator ()  ( const idx_t i, const idx_t j ) const noexcept
    {
        HLRCOMPRESS_DBG_ASSERT( i < idx_t(_length[0]) && j < idx_t(_length[1]) );
        return super_t::_data[ j * _stride[1] + i * _stride[0] ];
    }

    // return reference to coefficient (i,j)
    value_t &    operator ()  ( const idx_t i, const idx_t j ) noexcept
    {
        HLRCOMPRESS_DBG_ASSERT( i < idx_t(_length[0]) && j < idx_t(_length[1]) );
        return super_t::_data[ j * _stride[1] + i * _stride[0] ];
    }

    // return pointer to internal data
    value_t *    data         () const noexcept { return super_t::_data; }

    // return stride w.r.t. row index set
    size_t       row_stride   () const noexcept { return _stride[0]; }

    // return stride w.r.t. column index set
    size_t       col_stride   () const noexcept { return _stride[1]; }

    // optimised resize: only change if (n,m) != (nrows,ncols)
    void         resize       ( const size_t  n,
                                const size_t  m )
    {
        if (( _length[0] != n ) || ( _length[1] != m ))
        {
            *this = std::move( matrix( n, m ) );
        }// if
    }
    
    //
    // construction operators
    //

    // create real copy of matrix
    matrix< value_t >  copy () const
    {
        matrix< value_t >  M( *this, copy_value );

        return M;
    }
    
    // create reference to this matrix
    matrix< value_t >  reference () const
    {
        matrix< value_t >  M( *this, copy_reference );

        return M;
    }
    
    // return matrix referencing sub matrix defined by \a r1 × \a r2
    matrix< value_t >  operator () ( const range & r1, const range & r2 ) const
    {
        return matrix< value_t >( *this, r1, r2 );
    }
                          
    // return vector referencing column j
    vector< value_t >  column   ( const idx_t  j ) const
    {
        return vector< value_t >( *this, range::all, j );
    }
    
    // return vector referencing row i
    vector< value_t >  row      ( const idx_t  i ) const
    {
        return vector< value_t >( *this, i, range::all );
    }

    // return size in bytes used by this object
    size_t  byte_size () const
    {
        return sizeof( value_t ) * _length[0] * _length[1] + sizeof(_length) + sizeof(_stride);
    }
};

//
// gives access to matrix value type
//
template < typename T_value >
struct matrix_value_type< matrix< T_value > >
{
    using  value_t = T_value;
};

//
// signals, that T is of matrix type
//
template < typename T >
struct is_matrix< matrix< T > >
{
    static const bool  value = true;
};

}}// namespace hlrcompress::blas

//
// include matrix views
//
#include <hlrcompress/blas/matrix_view.hh>

#endif  // __HLRCOMPRESS_BLAS_MATRIX_HH
