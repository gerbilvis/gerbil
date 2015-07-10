#include "gerbilapplication.h"

#include <iostream>
#include <string>


int main(int argc, char **argv)
{
	QCoreApplication::setOrganizationName("FAU");
	QCoreApplication::setOrganizationDomain("fau.de");
	QCoreApplication::setApplicationName("Gerbil");


	GerbilApplication app(argc, argv);
	app.run();

	// never reached, GerbilApplication::run() does not return.
	return EXIT_FAILURE;
}

