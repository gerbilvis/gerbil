#ifndef VISE_DATA_TRAITS_H
#define VISE_DATA_TRAITS_H

#include <climits>
#include <cfloat>
#include <cstring>

template <typename DataType> struct DataTraits {
};

template <>
struct DataTraits<float> {
	typedef float data_type;
	typedef float aggregate_type;
	static inline data_type maxValue() { return FLT_MAX; }
	static inline data_type minValue() { return FLT_MIN; }
	static inline data_type epsilon() { return FLT_EPSILON; }
	static inline bool is_floating() { return true; }
};

template <>
struct DataTraits<double> {
	typedef double data_type;
	typedef double aggregate_type;
	static inline data_type maxValue() { return DBL_MAX; }
	static inline data_type minValue() { return DBL_MIN; }
	static inline data_type epsilon() { return DBL_EPSILON; }
	static inline bool is_floating() { return true; }
};

template <>
struct DataTraits<int> {
	typedef int data_type;
	typedef long int aggregate_type;
	static inline data_type maxValue() { return INT_MAX; }
	static inline data_type minValue() { return INT_MIN; }
	static inline data_type epsilon() { return 1; }
	static inline bool is_floating() { return false; }
};

template <>
struct DataTraits<short int> {
	typedef short int data_type;
	typedef int aggregate_type;
	static inline data_type maxValue() { return SHRT_MAX; }
	static inline data_type minValue() { return SHRT_MIN; }
	static inline data_type epsilon() { return 1; }
	static inline bool is_floating() { return false; }
};

template <>
struct DataTraits<char> {
	typedef char data_type;
	typedef short int aggregate_type;
	static inline data_type maxValue() { return CHAR_MAX; }
	static inline data_type minValue() { return CHAR_MIN; }
	static inline data_type epsilon() { return 1; }
	static inline bool is_floating() { return false; }
};

template<>
struct DataTraits<unsigned int> {
	typedef unsigned int data_type;
	typedef unsigned long int aggregate_type;
	static inline data_type maxValue() { return UINT_MAX; }
	static inline data_type minValue() { return 0; }
	static inline data_type epsilon() { return 1; }
	static inline bool is_floating() { return false; }
};

template<>
struct DataTraits<unsigned short int> {
	typedef unsigned short int data_type;
	typedef unsigned int aggregate_type;
	static inline data_type maxValue() { return USHRT_MAX; }
	static inline data_type minValue() { return 0; }
	static inline data_type epsilon() { return 1; }
	static inline bool is_floating() { return false; }
};

template<>
struct DataTraits<unsigned char> {
	typedef unsigned char data_type;
	typedef unsigned short int aggregate_type;
	static inline data_type maxValue() { return UCHAR_MAX; }
	static inline data_type minValue() { return 0; }
	static inline data_type epsilon() { return 1; }
	static inline bool is_floating() { return false; }
};

template<typename T>
int cmp(T a, T b) {
	T tmp = a - b;
	if (fabs(tmp) <= DataTraits<T>::epsilon() ) {
		return 0;
	}
	return (tmp > 0) ? 1 : -1;
}


//
#endif // DATATRAITS_H
