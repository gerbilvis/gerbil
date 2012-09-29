#ifndef SUPERPIXELS_COMMANDS_COMMAND_GRIDSEG_H
#define SUPERPIXELS_COMMANDS_COMMAND_GRIDSEG_H

#include "command.h"
#include "gridseg_config.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace vole {

class CommandGridSeg : public Command {
public:
	CommandGridSeg();

public:
	int execute();
	void printShortHelp() const;
	void printHelp() const;

protected:
	cv::Mat_<cv::Vec3b> fuse_by_segmentation(cv::Mat_<cv::Vec3b> prior_segmentation, int step_size_x, int step_size_y, int n_blocks_x, int n_blocks_y);

	GridSegConfig config;
};

} // vole

#endif // SUPERPIXELS_COMMANDS_COMMAND_GRIDSEG_H
