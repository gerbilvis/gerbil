#ifndef GERBIL_CPLUSPLUS_H
#define GERBIL_CPLUSPLUS_H

/** \file gerbil_cplusplus.h
 *  This file contains macros for C++11 and future C++ standards
 *  compatibility.
 */

#define GBL_TO_STR(x)   GBL_TO_XSTR(x)
#define GBL_TO_XSTR(x)  #x
#define GBL_FALSE       0

// FIXME Allegedly boost has macros providing override and final depending on
// compiler support. Use this instead.

// GCC/g++ sets __cplusplus >= 201103 since version 4.7 iff compiling with
// -std=c++11.
// MSVC++ 11.0 _MSC_VER == 1700 (Visual Studio 2012)

// Tested with g++ (Ubuntu 4.8.2-19ubuntu1) 4.8.2
// TODO Test this with MSVC (remove GBL_FALSE and commit)
#if GBL_FALSE && (__cplusplus >= 201103 ||  MSC_VER >= 1700)
#define GBL_OVERRIDE override
#define GBL_FINAL    final
#define GBL_NULLPTR  nullptr
#else
#define GBL_OVERRIDE
#define GBL_FINAL
#define GBL_NULLPTR  0
#endif

#endif // GERBIL_CPLUSPLUS_H
