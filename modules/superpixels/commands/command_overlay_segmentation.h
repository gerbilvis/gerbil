#ifndef SUPERPIXELS_COMMAND_COMMAND_OVERLAY_SEGMENTATION_H
#define SUPERPIXELS_COMMAND_COMMAND_OVERLAY_SEGMENTATION_H

#include "command.h"
#include "overlay_segmentation_config.h"

namespace vole {

class CommandOverlaySegmentation : public Command {
public:
	CommandOverlaySegmentation();

public:
	int execute();
	void printShortHelp() const;
	void printHelp() const;

protected:
	OverlaySegmentationConfig config;
};

} // vole

#endif // SUPERPIXELS_COMMAND_COMMAND_OVERLAY_SEGMENTATION_H
