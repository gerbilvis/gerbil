#include "export.h"
#include "imginput.h"

namespace imginput {

int Export::execute()
{
	auto image = imginput::ImgInput(config).execute();
	image->write_out(config.output);
	return 0; // success
}

void Export::printShortHelp() const {
	std::cout << "Export image in Gerbil format." << std::endl;
}

void Export::printHelp() const {
	std::cout << "The output consists of a text file and directory of same name\n"
	             "that holds each band in a PNG file.";
	std::cout << std::endl;
}

}
