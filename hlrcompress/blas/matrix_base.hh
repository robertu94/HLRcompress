#ifndef __HLRCOMPRESS_BLAS_MATRIXBASE_HH
#define __HLRCOMPRESS_BLAS_MATRIXBASE_HH
//
// Project     : HLRcompress
// Module      : blas/matrix_base
// Description : base class for all matrices and matrix views
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hlrcompress/blas/range.hh>

namespace hlrcompress { namespace blas {

//
// gives access to value value type
//
template < typename matrix_t >
struct matrix_value_type;

template < typename matrix_t > using matrix_value_type_t = typename matrix_value_type< matrix_t >::value_t;


//
// signals, that T is of matrix type
//
template < typename matrix_t >
struct is_matrix
{
    static const bool  value = false;
};

template < typename matrix_t > inline constexpr bool is_matrix_v = is_matrix< matrix_t >::value;

//
// BLAS matrix views (only for BLAS/LAPACK functions)
//
enum blasview_t
{
    BLAS_NORMAL     = 'N',
    BLAS_TRANSPOSED = 'T',
    BLAS_ADJOINT    = 'C'
};

//
// matrix transformations
//
enum matop_t
{
    // do not change matrix
    apply_normal     = 'N',

    // use transposed matrix
    apply_transposed = 'T',

    // use adjoint, e.g. conjugate transposed matrix
    apply_adjoint    = 'C',

    // use conjugate matrix
    apply_conjugate  = 'R',
};

//
// defines basic interface for matrices
//
template < typename derived_t >
class matrix_base
{
public:
    // scalar value type of matrix
    using  value_t = matrix_value_type_t< derived_t >;

public:
    //
    // data access
    //

    // return number of rows of matrix
    size_t       nrows        () const noexcept { return derived().nrows(); }
    
    // return number of columns of matrix
    size_t       ncols        () const noexcept { return derived().ncols(); }

    // return number of rows of matrix
    range        row_range    () const noexcept { return range( 0, idx_t(nrows())-1 ); }
    
    // return number of columns of matrix
    range        col_range    () const noexcept { return range( 0, idx_t(ncols())-1 ); }

    // return coefficient (i,j)
    value_t      operator ()  ( const idx_t i, const idx_t j ) const noexcept
    {
        return derived()(i,j);
    }

    // return reference to coefficient (i,j)
    value_t &    operator ()  ( const idx_t i, const idx_t j ) noexcept
    {
        return derived()(i,j);
    }

    // return pointer to internal data
    value_t *    data         () const noexcept { return derived().data(); }

    // return stride w.r.t. row index set
    size_t       row_stride   () const noexcept { return derived().row_stride(); }

    // return stride w.r.t. column index set
    size_t       col_stride   () const noexcept { return derived().col_stride(); }

    // return BLAS matrix view of matrix object
    blasview_t   blas_view    () const noexcept { return derived().blas_view(); }

    // return number of rows of actual BLAS matrix
    size_t       blas_nrows   () const noexcept { return derived().blas_nrows(); }

    // return number of columns of actual BLAS matrix
    size_t       blas_ncols   () const noexcept { return derived().blas_ncols(); }

private:
    // convert to derived type
    derived_t &        derived  ()       noexcept { return * static_cast<       derived_t * >( this ); }
    const derived_t &  derived  () const noexcept { return * static_cast< const derived_t * >( this ); }
};

//
// signals, that T is of matrix type
//
template < typename matrix_t >
struct is_matrix< matrix_base< matrix_t > >
{
    static const bool  value = is_matrix< matrix_t >::value;
};

}}// namespace hlrcompress::blas

#endif  // __HLRCOMPRESS_BLAS_MATRIXBASE_HH
