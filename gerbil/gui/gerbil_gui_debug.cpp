#include "gerbil_gui_debug.h"

#include <iostream>

#ifdef __GNUC__
#include <cxxabi.h>
#endif /* __GNUC__ */

#include <stdlib.h>
#include <string.h>

void ggdb_print_method(const char *clsname, const char *funname)
{
	int status = -1;
#ifdef __GNUC__
	char *dmname = abi::__cxa_demangle(clsname, NULL, NULL, &status);
	if(dmname) {
		// for some reason g++ appends a '*' to the class name, removed here
		int len = strlen(dmname);
		dmname[len-1 >= 0 ? len-1 : 0 ] = '\0';
		if(0 == status) {
			std::cerr << dmname << "::" << funname << "()";
		}
		free(dmname);
	}
#endif /* __GNUC__ */
	if(0 != status){ /* fall back to mangled name */
		std::cerr << clsname << "::" << funname;
	}
}
