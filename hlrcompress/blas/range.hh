#ifndef __HLRCOMPRESS_BLAS_RANGE_HH
#define __HLRCOMPRESS_BLAS_RANGE_HH
//
// Project     : HLRcompress
// Module      : blas/range
// Description : modifies indexset into Matlab range
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hlrcompress/hlr/indexset.hh>

namespace hlrcompress { namespace blas {

//
// indexset with modified ctors to act like in Matlab
//
class range : public indexset
{
public:
    //
    // constructors
    // - last < first => indexset = ∅
    //

    // create index set { \a afirst ... \a last }
    range ( const idx_t   afirst,
            const idx_t   alast ) noexcept
            : indexset( afirst, alast )
    {}

    // create index set { \a pos }
    range ( const idx_t  pos ) noexcept
            : indexset( pos, pos )
    {}

    // copy constructor for indexset objects
    range ( const indexset &  is ) noexcept
            : indexset( is )
    {}
            
    //
    // indicates full range
    //
    
    static range  all;
};

// instantiation of range::all
inline range  range::all( -1, -1 );

}}// namespace hlrcompress::blas

#endif  // __HLRCOMPRESS_BLAS_RANGE_HH
