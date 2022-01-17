#ifndef __HLRCOMPRESS_INDEXSET_HH
#define __HLRCOMPRESS_INDEXSET_HH
//
// Project     : HLRcompress
// Module      : indexset
// Description : represents and indexset
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <string>

namespace hlrcompress
{

// basic index type
using idx_t = long;

struct indexset
{
private:
    // first/last index in the set
    idx_t  _first, _last;

public:
    ////////////////////////////////////////////////////////
    //
    // ctors
    //

    // construct empty index set
    indexset () noexcept
    {
        _first = 0;
        _last  = -1;
    }

    // construct indexset if size n
    explicit
    indexset ( const size_t  n ) noexcept
    {
        _first = 0;
        _last  = idx_t(n)-1;
    }

    // construct indexset by first and last index
    indexset ( const idx_t  afirst,
               const idx_t  alast ) noexcept
    {
        _first = afirst;
        _last  = alast;
    }

    // copy constructor
    indexset ( const indexset & is ) noexcept
    {
        *this = is;
    }
    
    ////////////////////////////////////////////////////////
    //
    // access indexset data
    //

    idx_t  first () const noexcept { return _first; }
    idx_t  last  () const noexcept { return _last;  }

    size_t size  () const noexcept
    {
        auto  n = _last - _first + 1;

        return n >= 0 ? n : 0;
    }
    
    ////////////////////////////////////////////////////////
    //
    // misc. methods
    //

    // copy operator
    indexset & operator = ( const indexset & is ) noexcept
    {
        _first = is._first;
        _last  = is._last;

        return *this;
    }
    
    // equality operator
    bool  operator == ( const indexset & is ) const noexcept
    {
        return ((_first == is._first) && (_last == is._last));
    }
    
    // inequality operator
    bool  operator != ( const indexset & is ) const noexcept
    {
        return ((_first != is._first) || (_last != is._last));
    }
    
    // string output
    std::string  to_string () const
    {
        if ( size() == 0 ) return "{}";
        if ( size() == 1 ) return "{" + std::to_string( _first ) + "}";
        else               return "{" + std::to_string( _first ) + ":" + std::to_string( _last ) + "}";
    }

    // return size in bytes used by this object
    size_t  byte_size () const { return 2*sizeof(idx_t); }
};

//////////////////////////////////////////////////////////
//
// functional ctors
//

inline
indexset
is ( const idx_t  first,
     const idx_t  last ) noexcept
{
    return indexset( first, last );
}

//////////////////////////////////////////////////////////
//
// arithmetics for indexsets
//

//
// add offset to indexset
//
inline
indexset
operator + ( const indexset &  is,
             const idx_t       ofs ) noexcept
{
    return indexset( is.first() + ofs, is.last() + ofs );
}

//
// subtract offset from indexset
//
inline
indexset
operator - ( const indexset &  is,
             const idx_t       ofs ) noexcept
{
    return indexset( is.first() - ofs, is.last() - ofs );
}

//////////////////////////////////////////////////////////
//
// misc.
//

//
// conversion to string
//
std::string
to_string ( const indexset &  is )
{
    return is.to_string();
}

}// namespace hlrcompress

#endif  // __HLRCOMPRESS_INDEXSET_HH
