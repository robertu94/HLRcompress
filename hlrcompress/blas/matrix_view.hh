#ifndef __HLRCOMPRESS_BLAS_MATRIX_VIEW_HH
#define __HLRCOMPRESS_BLAS_MATRIX_VIEW_HH
//
// Project     : HLRcompress
// Module      : blas/matrix_view
// Description : provides transposed and adjoint matrix views
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hlrcompress/misc/type_traits.hh>
#include <hlrcompress/blas/matrix_base.hh>

namespace hlrcompress { namespace blas  {

//
// return conjugate of given matrix operation
//
inline
matop_t
conjugate ( const matop_t   op )
{
    switch ( op )
    {
        case apply_normal     : return apply_conjugate;
        case apply_conjugate  : return apply_normal;
        case apply_transposed : return apply_adjoint ;
        case apply_adjoint    : return apply_transposed;
        default               : HLRCOMPRESS_ERROR( "unknown matrix operation" );
    }// switch
}

//
// return transposed of given matrix operation
//
inline
matop_t
transposed ( const matop_t   op )
{
    switch ( op )
    {
        case apply_normal     : return apply_transposed ;
        case apply_conjugate  : return apply_adjoint ;
        case apply_transposed : return apply_normal ;
        case apply_adjoint    : return apply_conjugate;
        default               : HLRCOMPRESS_ERROR( "unknown matrix operation" );
    }// switch
}

//
// return adjoint of given matrix operation
//
inline
matop_t
adjoint ( const matop_t   op )
{
    switch ( op )
    {
        case apply_normal     : return apply_adjoint;
        case apply_conjugate  : return apply_transposed;
        case apply_transposed : return apply_conjugate;
        case apply_adjoint    : return apply_normal;
        default               : HLRCOMPRESS_ERROR( "unknown matrix operation" );
    }// switch
}

//
// Provide transposed view of a matrix.
//
template < typename matrix_t >
class transpose_view : public matrix_base< transpose_view< matrix_t > >
{
public:
    using  value_t = typename matrix_t::value_t;

private:
    const matrix_t &  _mat;

public:
    transpose_view ( const matrix_t &  M ) noexcept
            : _mat( M )
    {}

    value_t *        data         () const noexcept { return _mat.data(); }

    size_t           nrows        () const noexcept { return _mat.ncols(); }
    size_t           ncols        () const noexcept { return _mat.nrows(); }

    size_t           blas_nrows   () const noexcept { return _mat.nrows(); }
    size_t           blas_ncols   () const noexcept { return _mat.ncols(); }

    size_t           row_stride   () const noexcept { return _mat.row_stride(); }
    size_t           col_stride   () const noexcept { return _mat.col_stride(); }

    blasview_t       blas_view    () const
    {
        switch ( _mat.blas_view() )
        {
        case BLAS_NORMAL     :
            return BLAS_TRANSPOSED;
            
        case BLAS_TRANSPOSED :
            return BLAS_NORMAL;
            
        case BLAS_ADJOINT    :
            // just conjugate without transpose is not possible with standard BLAS/LAPACK
            if ( is_complex_type_v< value_t > )
                HLRCOMPRESS_ERROR( "not yet implemented" )
            
            return BLAS_NORMAL;
        }// switch

        return BLAS_TRANSPOSED;
    }
    
    value_t    operator ()  ( const idx_t i, const idx_t j ) const noexcept { return _mat( j, i ); }
};

//
// gives access to matrix value type
//
template < typename matrix_t >
struct matrix_value_type< transpose_view< matrix_t > >
{
    using  value_t = value_type_t< matrix_t >;
};

//
// signals, that T is of matrix type
//
template < typename T >
struct is_matrix< transpose_view< T > >
{
    static const bool  value = true;
};

//
// return transposed view object for matrix
//
template < typename T >
transpose_view< T >
transposed ( const T &  M )
{
    return transpose_view< T >( M );
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Provide adjoint view, e.g. conjugate transposed of a given matrix.
//
template < typename matrix_t >
class adjoin_view : public matrix_base< adjoin_view< matrix_t > >
{
public:
    using  value_t = typename matrix_t::value_t;

private:
    const matrix_t &  _mat;

public:
    adjoin_view ( const matrix_t &  M ) noexcept
            : _mat( M )
    {}

    value_t *        data         () const noexcept { return _mat.data(); }

    size_t           nrows        () const noexcept { return _mat.ncols(); }
    size_t           ncols        () const noexcept { return _mat.nrows(); }

    size_t           blas_nrows   () const noexcept { return _mat.nrows(); }
    size_t           blas_ncols   () const noexcept { return _mat.ncols(); }

    size_t           row_stride   () const noexcept { return _mat.row_stride(); }
    size_t           col_stride   () const noexcept { return _mat.col_stride(); }

    blasview_t       blas_view    () const
    {
        switch ( _mat.blas_view() )
        {
        case BLAS_NORMAL     :
            return BLAS_ADJOINT;
            
        case BLAS_TRANSPOSED :
            // just conjugate without transpose is not possible with standard BLAS/LAPACK
            if ( is_complex_type_v< value_t > )
                HLRCOMPRESS_ERROR( "not yet implemented" )
            
            return BLAS_NORMAL;
            
        case BLAS_ADJOINT    :
            return BLAS_NORMAL;
        }// switch

        return BLAS_ADJOINT;
    }
    
    value_t    operator ()  ( const idx_t i, const idx_t j ) const noexcept
    {
        if constexpr( is_complex_type_v< value_t > ) return std::conj( _mat( j, i ) );
        else                                         return _mat( j, i );
    }
};

//
// gives access to matrix value type
//
template < typename matrix_t >
struct matrix_value_type< adjoin_view< matrix_t > >
{
    using  value_t = value_type_t< matrix_t >;
};

//
// signals, that T is of matrix type
//
template < typename T >
struct is_matrix< adjoin_view< T > >
{
    static const bool  value = true;
};

//
// return adjoint view object for matrix
//
template < typename T >
adjoin_view< T >
adjoint ( const T &  M )
{
    return adjoin_view< T >( M );
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Provide generic view to a matrix, e.g. transposed or adjoint.
//
template < typename matrix_t >
class matrix_view : public matrix_base< matrix_view< matrix_t > >
{
public:
    using  value_t = typename matrix_t::value_t;

private:
    const matop_t     _op;
    const matrix_t &  _mat;

public:
    matrix_view ( const matop_t     op,
                 const matrix_t &  M ) noexcept
            : _op(op), _mat( M )
    {}

    value_t *        data         () const noexcept { return _mat.data(); }

    size_t           nrows        () const noexcept
    {
        switch ( _op )
        {
            case apply_normal     :
            case apply_conjugate  : return _mat.nrows();
            case apply_transposed :
            case apply_adjoint    : return _mat.ncols();
        }// switch

        return _mat.nrows();
    }

    size_t           ncols        () const noexcept
    {
        switch ( _op )
        {
            case apply_normal     :
            case apply_conjugate  : return _mat.ncols();
            case apply_transposed :
            case apply_adjoint    : return _mat.nrows();
        }// switch

        return _mat.ncols();
    }

    size_t           blas_nrows   () const noexcept { return _mat.nrows(); }
    size_t           blas_ncols   () const noexcept { return _mat.ncols(); }

    size_t           row_stride   () const noexcept { return _mat.row_stride(); }
    size_t           col_stride   () const noexcept { return _mat.col_stride(); }

    blasview_t       blas_view    () const
    {
        if ( _op == apply_normal )
        {
            return _mat.blas_view();
        }// if
        else if ( _op == apply_conjugate )
        {
            switch ( _mat.blas_view() )
            {
                case BLAS_NORMAL     :
                    if ( is_complex_type_v< value_t > )
                        HLRCOMPRESS_ERROR( "not yet implemented" )
                    return BLAS_NORMAL;
            
                case BLAS_TRANSPOSED :
                    return BLAS_ADJOINT;
            
                case BLAS_ADJOINT    :
                    return BLAS_TRANSPOSED;
            }// switch
        }// if
        else if ( _op == apply_transposed )
        {
            switch ( _mat.blas_view() )
            {
                case BLAS_NORMAL     :
                    return BLAS_TRANSPOSED;
            
                case BLAS_TRANSPOSED :
                    return BLAS_NORMAL;
            
                case BLAS_ADJOINT    :
                    // just conjugate without transpose is not possible with standard BLAS/LAPACK
                    if ( is_complex_type_v< value_t > )
                        HLRCOMPRESS_ERROR( "not yet implemented" )
            
                    return BLAS_NORMAL;
            }// switch
        }// if
        else if ( _op == apply_adjoint )
        {
            switch ( _mat.blas_view() )
            {
                case BLAS_NORMAL     :
                    return BLAS_ADJOINT;
            
                case BLAS_TRANSPOSED :
                    // just conjugate without transpose is not possible with standard BLAS/LAPACK
                    if ( is_complex_type_v< value_t > )
                        HLRCOMPRESS_ERROR( "not yet implemented" )
            
                    return BLAS_NORMAL;
            
                case BLAS_ADJOINT    :
                    return BLAS_NORMAL;
            }// switch
        }// else
        
        return BLAS_NORMAL;
    }
    
    value_t    operator ()  ( const idx_t i, const idx_t j ) const noexcept
    {
        switch ( _op )
        {
            case apply_normal  :
                return _mat( i, j );
                
            case apply_conjugate :
                if constexpr( is_complex_type_v< value_t > ) return std::conj( _mat( i, j ) );
                else                                         return _mat( i, j );
                
            case apply_transposed   :
                return _mat( j, i );
                
            case apply_adjoint : 
                if constexpr( is_complex_type_v< value_t > ) return std::conj( _mat( j, i ) );
                else                                         return _mat( j, i );
        }// switch

        return _mat( i, j );
    }
};

//
// gives access to matrix value type
//
template < typename matrix_t >
struct matrix_value_type< matrix_view< matrix_t > >
{
    using  value_t = value_type_t< matrix_t >;
};

//
// signals, that T is of matrix type
//
template < typename matrix_t >
struct is_matrix< matrix_view< matrix_t > >
{
    static const bool  value = true;
};

//
// convert matrix M with op into view object
//
template < typename T >
matrix_view< T >
mat_view ( const matop_t  op,
           const T &      M )
{
    return matrix_view< T >( op, M );
}

}}// namespace hlrcompress::blas

#endif  // __HLRCOMPRESS_BLAS_MATRIX_VIEW_HH
