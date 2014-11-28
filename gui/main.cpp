#include <gerbil_cplusplus.h>
#include "gerbilapplication.h"

#include <iostream>
#include <string>

void dump_compiler_cplusplus_info()
{
	using namespace std;
	cout << "=========================================================" << endl;
	cout << "C++ Compiler Standard Compatibility Information" << endl;
	cout << "=========================================================" << endl;
	cout << "C++ version reported by compiler:           "
		 << __cplusplus << endl;
	cout << "Compiled with C++11 override keyword:       "
		 << ((!string(GBL_TO_STR(GBL_OVERRIDE)).empty()) ? "yes":"no") << endl;
	cout << "Compiled with C++11 final keyword:          "
		 << ((!string(GBL_TO_STR(GBL_FINAL)).empty()) ? "yes":"no") << endl;
}

int main(int argc, char **argv)
{
	QCoreApplication::setOrganizationName("FAU");
	QCoreApplication::setOrganizationDomain("fau.de");
	QCoreApplication::setApplicationName("Gerbil");

	// FIXME Remove for RELEASE
	dump_compiler_cplusplus_info();

	GerbilApplication app(argc, argv);
	app.run();

	// never reached, GerbilApplication::run() does not return.
	return EXIT_FAILURE;
}

