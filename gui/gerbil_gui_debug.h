#ifndef GERBIL_GUI_DEBUG_H
#define GERBIL_GUI_DEBUG_H

#include <string>

/*! \brief Append <class>::<method>() to cerr (no return type, no parameters)
 *
 * Tested on gcc 4.4.5 x64 (debian)
 */
void ggdb_print_method(const char *clsname, const char *funname);
std::string ggdb_method_string(const char *clsname, const char *funname);

// FIXME this should be configurable by CMAKE
#define GGDBG

// Gerbil Gui DeBuG
#ifdef GGDBG

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
class GGDBGEnterLeavePrint {
	std::string method_string;
public:
	GGDBGEnterLeavePrint(std::string method_string);
	~GGDBGEnterLeavePrint();
};

#define GGDBG_ENTER_LEAVE() \
	GGDBGEnterLeavePrint _ggdbg_enterLeaveObj(ggdb_method_string(typeid(this).name(), __func__))

#else /* GGDBG */

#define GGDBG_PRINT_METHOD()
#define GGDBG_CALL()
#define GGDBG_ENTER_LEAVE()
#define GGDBGM(expr)

class GGDBGLeavePrint {
	// does nothing
};
#endif /* GGDBG */

#endif // GERBIL_GUI_DEBUG_H
