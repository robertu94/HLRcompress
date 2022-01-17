#ifndef __HLRCOMPRESS_MISC_TYPE_TRAITS_HH
#define __HLRCOMPRESS_MISC_TYPE_TRAITS_HH
//
// Project     : HLRcompress
// Module      : misc/type_traits
// Description : defines various type traits
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <complex>

namespace hlrcompress
{

//
// just access internal value type
//
template <typename T> using value_type_t = typename T::value_t;

//
// provide real valued type forming base of T
//
template <typename T> struct real_type                       { using  type_t = T; };
template <typename T> struct real_type< std::complex< T > >  { using  type_t = T; };

template <typename T> using real_type_t = typename real_type< T >::type_t;

//
// signals complex valued types
//
template <typename T> struct is_complex_type                            { static constexpr bool value = false; };
template <>           struct is_complex_type< std::complex< float > >   { static constexpr bool value = true; };
template <>           struct is_complex_type< std::complex< double > >  { static constexpr bool value = true; };

template <typename T> inline constexpr bool is_complex_type_v = is_complex_type< T >::value;

}// namespace hlrcompress

#endif // __HLRCOMPRESS_MISC_TYPE_TRAITS_HH
