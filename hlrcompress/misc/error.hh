#ifndef __HLRCOMPRESS_MISC_ERROR_HH
#define __HLRCOMPRESS_MISC_ERROR_HH
//
// Project     : HLRcompress
// Module      : misc/error
// Description : error handling functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <iostream>
#include <string>

namespace hlrcompress
{

//
// breakpoint function as entry point for debugging
//
inline
void
breakpoint ()
{
    ;
}

//
// logging function
//
template < typename msg_t >
void
error ( const msg_t &  msg )
{
    std::cout << msg << std::endl;
    std::exit( 1 );
}

// throw exception with file info
#define HLRCOMPRESS_ERROR( msg )                                        \
    {                                                                   \
        hlrcompress::breakpoint();                                      \
        throw std::runtime_error( std::string( __FILE__ ) + ":" +       \
                                  std::to_string( __LINE__ ) +          \
                                  " : " + msg );                        \
    }

// always-on-assert
#define HLRCOMPRESS_ASSERT( expr )                                      \
    if ( ! ( expr ) )                                                   \
    {                                                                   \
        hlrcompress::breakpoint();                                      \
        HLRCOMPRESS_ERROR( ( std::string( #expr ) + " failed" ) );      \
    }

// debug assert
#if defined(NDEBUG)
#  define HLRCOMPRESS_DBG_ASSERT( expr )
#else
#  define HLRCOMPRESS_DBG_ASSERT( expr ) HLRCOMPRESS_ASSERT( expr )
#endif

}// namespace hlrcompress

#endif // __HLRCOMPRESS_MISC_ERROR_HH
