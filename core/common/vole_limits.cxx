#include "vole_limits.h"

template<> signed short VoleLimits::max<signed short>() { return SHRT_MAX; }
template<> signed short VoleLimits::min<signed short>() { return SHRT_MIN; }
template<> unsigned short VoleLimits::max<unsigned short>() { return USHRT_MAX; }
template<> unsigned short VoleLimits::min<unsigned short>() { return 0; }

template<> signed int VoleLimits::max<signed int>() { return INT_MAX; }
template<> signed int VoleLimits::min<signed int>() { return INT_MIN; }
template<> unsigned int VoleLimits::max<unsigned int>() { return UINT_MAX; }
template<> unsigned int VoleLimits::min<unsigned int>() { return 0; }

template<> signed long VoleLimits::max<signed long>() { return LONG_MAX; }
template<> signed long VoleLimits::min<signed long>() { return LONG_MIN; }
template<> unsigned long VoleLimits::max<unsigned long>() { return ULONG_MAX; }
template<> unsigned long VoleLimits::min<unsigned long>() { return 0; }

template<> signed long long VoleLimits::max<signed long long>() { return LLONG_MAX; }
template<> signed long long VoleLimits::min<signed long long>() { return LLONG_MIN; }
template<> unsigned long long VoleLimits::max<unsigned long long>() { return ULLONG_MAX; }
template<> unsigned long long VoleLimits::min<unsigned long long>() { return 0; }

// FIXME: Warum soll float & double nicht gehen?
template<> float VoleLimits::max<float>() { return FLT_MAX; }
template<> float VoleLimits::min<float>() { return FLT_MIN; }

template<> double VoleLimits::max<double>() { return DBL_MAX; }
template<> double VoleLimits::min<double>() { return DBL_MIN; }


