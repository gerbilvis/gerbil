#ifndef GERBIL_GUI_DEBUG_H
#define GERBIL_GUI_DEBUG_H

/*! \brief Append <class>::<method>() to cerr (no return type, no parameters)
 *
 * Tested on gcc 4.4.5 x64 (debian)
 */
void ggdb_print_method(const char *clsname, const char *funname);

// FIXME this should be configurable by CMAKE
//#define GGDBG

// Gerbil Gui DeBuG
#ifdef GGDBG

#include <boost/format.hpp>

#define GGDBG_PRINT_METHOD() ggdb_print_method(typeid(this).name(), __func__)

// print class name and function to cerr
#define GGDBG_CALL()  GGDBG_PRINT_METHOD(); std::cerr << std::endl;

// Gerbil Gui DeBuG class Method
// Append class name and method followed by expr to cerr.
#define GGDBGM(expr) \
	{ \
		using namespace std; \
		using namespace boost; \
		GGDBG_PRINT_METHOD(); \
		std::cerr << " " << expr; \
	}

#else /* GGDBG */

#define GGDBG_PRINT_METHOD()
#define GGDBG_CALL()
#define GGDBGM(expr)

#endif /* GGDBG */

#endif // GERBIL_GUI_DEBUG_H
