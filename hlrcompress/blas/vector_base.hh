#ifndef __HLRCOMPRESS_BLAS_VECTORBASE_HH
#define __HLRCOMPRESS_BLAS_VECTORBASE_HH
//
// Project     : HLRcompress
// Module      : blas/vector.hh
// Description : defines basic interface for vectors
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hlrcompress/blas/range.hh>
#include <hlrcompress/misc/type_traits.hh>

namespace hlrcompress { namespace blas {

//
// gives access to vector value type
//
template < typename vector_t >
struct vector_value_type;

template < typename vector_t > using vector_value_type_t = typename vector_value_type< vector_t >::value_t;

//
// signals, that vector_t is of vector type
//
template < typename vector_t >
struct is_vector
{
    static const bool  value = false;
};

template < typename vector_t > inline constexpr bool is_vector_v = is_vector< vector_t >::value;

//
// defines basic interface for vectors
//
template < typename derived_t >
class vector_base
{
public:
    // scalar value type of vector
    using  value_t = vector_value_type_t< derived_t >;

public:

    // return length of vector
    size_t    length      ()                  const noexcept { return derived().length(); }

    // return stride of index set
    size_t    stride      ()                  const noexcept { return derived().length(); }

    // return coefficient at position \a i
    value_t   operator () ( const idx_t   i ) const noexcept { return derived()( i ); }
    
    // return reference to coefficient at position \a i
    value_t & operator () ( const idx_t   i )       noexcept { return derived()( i ); }
    
    // give access to internal data
    value_t * data        ()                  const noexcept { return derived().data(); }

private:
    // convert to derived type
    derived_t &        derived  ()       noexcept { return * static_cast<       derived_t * >( this ); }
    const derived_t &  derived  () const noexcept { return * static_cast< const derived_t * >( this ); }
};

//
// vector type defined by derived type
//
template < typename vector_t >
struct is_vector< vector_base< vector_t > >
{
    static const bool  value = is_vector< vector_t >::value;
};

}}// namespace hlrcompress::blas

#endif  // __HLRCOMPRESS_BLAS_VECTORBASE_HH
