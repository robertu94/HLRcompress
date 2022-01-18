#ifndef __HLRCOMPRESS_MISC_COMPRESSION_HH
#define __HLRCOMPRESS_MISC_COMPRESSION_HH
//
// Project     : HLRcompress
// Module      : utils/compression
// Description : compression related functions and types
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hlrcompress/config.h>

#if USE_ZFP == 1
#include <zfpcarray2.h>
#endif

//
// compression configuration type
//

namespace hlrcompress
{

#if USE_ZFP == 1
using  zconfig_t = zfp_config;
#else 
struct zconfig_t {};
#endif

}// namespace hlr

#endif // __HLRCOMPRESS_MISC_COMPRESSION_HH
