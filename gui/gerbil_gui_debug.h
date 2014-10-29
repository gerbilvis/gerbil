/*****************************************************************************
 * Gerbil GUI debug macros.
 *
 * This file provides simple debug macros for the gerbil GUI. String
 * formatting depends on boost::format, included in this file.
 * These macros generate code depending on two macro definitions:
 *
 *  GGDBG: If not defined no debug code is generated. This is currently
 *  defined in this file and thus has no effect.
 *  GGDBG_MODULE: Same as above, but needs to be defined prior to including
 *  this file. Thus allows enabling/disabling debug per compilation unit.
 *
 *  Class and function name demangling
 *  ----------------------------------
 *
 *  The defined macros try to demangle class and function names. This is
 *  currently implemented for GCC.
 *
 *
 * Macros
 * -------
 *
 * GGDBG_CALL()		Just print the class and function name, i.e.
 *                  Class::function()
 * GGDBGM(expr)     Forward expr to std::cerr like
 *                  std::cerr << expr;
 *                  prepended with the current class and function name.
 *                  Namespace std is imported by default for convenience. I.e.
 *                  it is possible to write
 *					GGDBGM("hello world" << endl);
 * GGDBG_ENTER_LEAVE()
 *                  prints "enter" and "leave" prepended by class and function
 *                  name when the macro is called (enter) and when the
 *                  function returns (leave). The latter is accomplished using
 *                  a RAII object which guaranteed to be destructed by the
 *                  compiler.
 *
 */
#ifndef GERBIL_GUI_DEBUG_H
#define GERBIL_GUI_DEBUG_H

#include <string>

// Helper functions used by GGDBG_PRINT_METHOD
void ggdb_print_method(const char *clsname, const char *funname);
std::string ggdb_method_string(const char *clsname, const char *funname);

// FIXME this should be configurable by CMAKE
#define GGDBG

// Gerbil Gui DeBuG
#if defined(GGDBG) && defined(GGDBG_MODULE)

#include <boost/format.hpp>

#define GGDBG_PRINT_METHOD() ggdb_print_method(typeid(this).name(), __func__)

// print class name and function to cerr
#define GGDBG_CALL()  GGDBG_PRINT_METHOD(); std::cerr << std::endl;

// also print viewer representation type
#define GGDBG_CALL_VT()  GGDBG_PRINT_METHOD(); std::cerr << " type=" << getType() << std::endl;

// Gerbil Gui DeBuG class Method
// Append class name and method followed by expr to cerr.
#define GGDBGM(expr) \
	{ \
		using namespace std; \
		using namespace boost; \
		GGDBG_PRINT_METHOD(); \
		std::cerr << " " << expr; \
		std::cerr.flush(); \
	}

// Gerbil Gui DeBuG Print
#define GGDBGP(expr) \
	{ \
		using namespace std; \
		using namespace boost; \
		std::cerr << " " << expr; \
		std::cerr.flush(); \
	}

#define GGDBGM_VT(expr) \
	{ \
		using namespace std; \
		using namespace boost; \
		GGDBG_PRINT_METHOD(); \
		std::cerr  << " type=" << getType() << " " << expr; \
		std::cerr.flush(); \
	}

#define GGDBG_ENTER_LEAVE() \
	GGDBGEnterLeavePrint _ggdbg_enterLeaveObj(ggdb_method_string(typeid(this).name(), __func__))

#else /* GGDBG */

#define GGDBG_PRINT_METHOD() do {} while (false)
#define GGDBG_CALL()         do {} while (false)
#define GGDBG_ENTER_LEAVE()  do {} while (false)
#define GGDBGM(expr)         do {} while (false)
#define GGDBGP(expr)         do {} while (false)
#define GGDBGM_VT(expr)      do {} while (false)

#endif /* GGDBG */

class GGDBGEnterLeavePrint {
	std::string method_string;
public:
	GGDBGEnterLeavePrint(std::string method_string);
	~GGDBGEnterLeavePrint();
};


#endif // GERBIL_GUI_DEBUG_H
