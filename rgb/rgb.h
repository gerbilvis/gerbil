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
	~RGB();
	int execute();
#ifdef WITH_BOOST
	std::map<std::string, boost::any> execute(std::map<std::string, boost::any> &input, vole::ProgressObserver *po);
#endif
	cv::Mat3f execute(const multi_img& src, vole::ProgressObserver *po = NULL);

	void printShortHelp() const;
	void printHelp() const;
	// Receive progress updates from internal som calculation
	void setSomProgress(int percent);
	void abort() { abortFlag = true; }
protected:
	cv::Mat3f executePCA(const multi_img& src);
#ifdef WITH_EDGE_DETECT
	cv::Mat3f executeSOM(const multi_img& src);
#endif

	bool progressUpdate(float percent, vole::ProgressObserver *po);
public:
	RGBConfig config;

private:
	multi_img *srcimg;

	// external observer from client : observes our progress
	vole::ProgressObserver *po;

	// internal observer: observe SOM training
	vole::ProgressObserver *calcPo;

	volatile bool abortFlag;
};

}

#endif
