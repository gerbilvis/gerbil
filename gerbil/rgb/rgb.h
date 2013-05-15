#ifndef RGB_H
#define RGB_H

#include "rgb_config.h"
#include <command.h>

namespace gerbil {

class RGB : public vole::Command {
public:
	RGB();
	int execute();
	cv::Mat3f execute(const multi_img& src);

	void printShortHelp() const;
	void printHelp() const;

protected:
	cv::Mat3f executePCA(const multi_img& src);
#ifdef WITH_EDGE_DETECT
	cv::Mat3f executeSOM(const multi_img& src);
#endif

public:
	RGBConfig config;
};

}

#endif
