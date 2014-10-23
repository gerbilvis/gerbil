#ifndef RGB_H
#define RGB_H

#include "rgb_config.h"
#include <command.h>
#include <progress_observer.h>
#ifdef WITH_BOOST
#include <boost/any.hpp>
#endif

#ifdef WITH_SOM
namespace som { // forward declaration for executeSOM
	class GenSOM;
}
#endif

namespace rgb {

class RGB : public shell::Command {
public:
	RGB();
	int execute();
#ifdef WITH_BOOST
	std::map<std::string, boost::any>
	execute(std::map<std::string, boost::any> &input,
			ProgressObserver *po);
#endif
	cv::Mat3f execute(const multi_img& src, ProgressObserver *po = NULL);

	void printShortHelp() const;
	void printHelp() const;

protected:
	cv::Mat3f executePCA(const multi_img& src, ProgressObserver *po);
#ifdef WITH_SOM
	cv::Mat3f executeSOM(const multi_img& src, ProgressObserver *po,
						 boost::shared_ptr<som::GenSOM> som
						 = boost::shared_ptr<som::GenSOM>());
#endif

public:
	RGBConfig config;
};

}

#endif
