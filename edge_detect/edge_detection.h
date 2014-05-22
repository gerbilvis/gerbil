#ifndef EDGE_DETECTION_H
#define EDGE_DETECTION_H

#include <command.h>
#include "edge_detection_config.h"

namespace vole {


class EdgeDetection : public  Command
{
public:
	EdgeDetection();
	virtual ~EdgeDetection();
	virtual int execute();

	void printShortHelp() const;
	void printHelp() const;


protected:
	vole::EdgeDetectionConfig config;
};

} // namespace vole

#endif // EDGE_DETECTION_H
