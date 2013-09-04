#include "gerbil_gui_debug.h"

#include <iostream>

#ifdef __GNUC__
#include <cxxabi.h>
#endif /* __GNUC__ */

#include <stdlib.h>
#include <string.h>

std::string ggdb_method_string(const char *clsname, const char *funname)
{
	using namespace std;
	string method_string;
	int status = -1;
#ifdef __GNUC__
	char *dmname = abi::__cxa_demangle(clsname, NULL, NULL, &status);
	if(dmname) {
		// for some reason g++ appends a '*' to the class name, removed here
		int len = strlen(dmname);
		dmname[len-1 >= 0 ? len-1 : 0 ] = '\0';
		if(0 == status) {
			method_string = string(dmname) + "::" + funname + "()";
		}
		free(dmname);
	}
#endif /* __GNUC__ */
	if(0 != status){ /* fall back to mangled name */
		method_string = string(clsname) + "::" + funname + "()";
	}
	return method_string;
}

void ggdb_print_method(const char *clsname, const char *funname)
{
	std::cerr << ggdb_method_string(clsname, funname);
}

// Gerbil Gui DeBuG
#ifdef GGDBG

GGDBGEnterLeavePrint::GGDBGEnterLeavePrint(std::string method_string)
	:method_string(method_string)
{
	std::cerr << method_string << " enter" << std::endl;
	std::cerr.flush();
}

GGDBGEnterLeavePrint::~GGDBGEnterLeavePrint()
{
	std::cerr << method_string << " leave" << std::endl;
	std::cerr.flush();
}

#endif /* GGDBG */
