#ifndef MEANSHIFT_SP_H
#define MEANSHIFT_SP_H

#include "meanshift_config.h"
#include <command.h>

namespace vole {

class MeanShiftSP : public Command {
public:
	MeanShiftSP();
	~MeanShiftSP();
	int execute();
	std::map<std::string, boost::any> execute(std::map<std::string, boost::any> &input, ProgressObserver *progress);
	cv::Mat1s execute(multi_img::ptr input, multi_img::ptr input_grad);

	void printShortHelp() const;
	void printHelp() const;

	MeanShiftConfig config;
};

}

#endif
