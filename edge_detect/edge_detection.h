#ifndef EDGE_DETECTION_H
#define EDGE_DETECTION_H

#include <command.h>
#include "edge_detection_config.h"

namespace edge_detect {


class EdgeDetection : public shell::Command
{
public:
	EdgeDetection();
	virtual ~EdgeDetection();
	virtual int execute();

	void printShortHelp() const;
	void printHelp() const;


protected:
	edge_detect::EdgeDetectionConfig config;
};

} // module namespace

#endif // EDGE_DETECTION_H
