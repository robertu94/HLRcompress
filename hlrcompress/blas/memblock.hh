#ifndef __HLRCOMPRESS_BLAS_MEMBLOCK_HH
#define __HLRCOMPRESS_BLAS_MEMBLOCK_HH
//
// Project     : HLRcompress
// Module      : blas/memblock
// Description : represents a general memory block
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <cstdlib>

#include <hlrcompress/misc/error.hh>

namespace hlrcompress { namespace blas {

//
// indicates copy policy
//
enum copy_policy_t
{
    copy_reference, // copy pointer to data only
    copy_value      // copy actual data
};

//
// general memory block with managed ownership
//
template < class T_value >
struct memblock
{
public:
    // internal value type
    using  value_t = T_value;

protected:
    // pointer of data
    value_t *   _data;

    // indicates, this object is owner of memory block
    bool        _is_owner;

public:
    //
    // constructors destructor
    //

    // ctor with nullptr data and 0 references
    memblock  () noexcept
            : _data( nullptr )
            , _is_owner( false )
    {}
    
    // ctor for n elements of value_t and 0 references
    memblock  ( const size_t  n )
            : _data( nullptr )
            , _is_owner( false )
    {
        alloc( n );
    }
    
    // copy ctor (copy reference!)
    memblock  ( const memblock &  b )
            : _data( b._data )
            , _is_owner( false )
    {}
    
    // move ctor (move ownership)
    memblock  ( memblock &&  b ) noexcept
            : _data( b._data )
            , _is_owner( b._is_owner )
    {
        b._is_owner = false;
    }

    // dtor removing all data
    ~memblock  ()
    {
        if ( _is_owner )
            delete[] _data;
    }

    // copy operator (copy reference!)
    memblock &  operator = ( const memblock &  b )
    {
        _data     = b._data;
        _is_owner = false;
        
        return *this;
    }

    // move ctor (move ownership)
    memblock &  operator = ( memblock &&  b ) noexcept
    {
        _data     = b._data;
        _is_owner = b._is_owner;  // only owner if b was owner
        
        b._is_owner = false;

        return *this;
    }

    //
    // initialise memory block
    //

    // initialise with raw memory pointer
    void init ( value_t *   ptr,
                const bool  ais_owner = false ) noexcept
    {
        if ( _is_owner )
            delete[] _data;

        _data     = ptr;
        _is_owner = ais_owner;
    }

    // initialise with memory block (copy OR move)
    void init ( memblock &  b,
                const bool  ais_owner = false )
    {
        if ( ais_owner && ! b._is_owner )
            HLRCOMPRESS_ERROR( "can not be owner of data NOT owned by given block" );
        
        if ( _is_owner )
            delete[] _data;

        _data     = b._data;
        _is_owner = ais_owner;

        if ( _is_owner )
            b._is_owner = false;
    }

    // allocate and initialise memory block
    void alloc ( const size_t   n,
                 const value_t  init_val = value_t(0) )
    {
        alloc_wo_value( n );

        for ( size_t  i = 0; i < n; i++ )
            _data[i] = init_val;
    }

    // only allocate memory block without intialisation
    void alloc_wo_value ( const size_t  n )
    {
        if ( _is_owner )
            delete[] _data;
        
        _data     = new value_t[n];
        _is_owner = true;
    }
    
    //
    // access data
    //

    // return pointer to internal array
    value_t *       data     ()       noexcept { return _data; }

    // return const pointer to internal array
    const value_t * data     () const noexcept { return _data; }

    // return is_owner status
    bool            is_owner () const noexcept { return _is_owner; }
};

}}// namespace hlrcompress::blas

#endif  // __HLRCOMPRESS_BLAS_MEMBLOCK_HH
