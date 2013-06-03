#ifndef RGB_H
#define RGB_H

#include "rgb_config.h"
#include <command.h>
#include <progress_observer.h> // TODO: raus

namespace gerbil {

class RGB : public vole::Command {
public:
	RGB();
	int execute();
#ifdef WITH_BOOST
	std::map<std::string, boost::any> execute(std::map<std::string, boost::any> &input, vole::ProgressObserver *po);
#endif
	cv::Mat3f execute(const multi_img& src, vole::ProgressObserver *po = NULL);

	void printShortHelp() const;
	void printHelp() const;

protected:
	cv::Mat3f executePCA(const multi_img& src, vole::ProgressObserver *po);
#ifdef WITH_EDGE_DETECT
	cv::Mat3f executeSOM(const multi_img& src, vole::ProgressObserver *po);
#endif

	bool progressUpdate(float percent, vole::ProgressObserver *po);

public:
	RGBConfig config;
};

}

#endif
