#ifndef RGB_H
#define RGB_H

#include "rgb_config.h"
#include <command.h>
#include <progress_observer.h>
#ifdef WITH_BOOST
#include <boost/any.hpp>
#endif

namespace gerbil {

class RGB : public vole::Command {
public:
	RGB();
	int execute();
#ifdef WITH_BOOST
	std::map<std::string, boost::any>
	execute(std::map<std::string, boost::any> &input,
			vole::ProgressObserver *po);
#endif
	cv::Mat3f execute(const multi_img& src, vole::ProgressObserver *po = NULL);

	void printShortHelp() const;
	void printHelp() const;

protected:
	cv::Mat3f executePCA(const multi_img& src, vole::ProgressObserver *po);
#ifdef WITH_SOM
	cv::Mat3f executeSOM(const multi_img& src, vole::ProgressObserver *po);
#endif

public:
	RGBConfig config;
};

}

#endif
