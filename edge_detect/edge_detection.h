#ifndef EDGE_DETECTION_H
#define EDGE_DETECTION_H

#include "som.h"
#include "edge_detection_config.h"
#include <command.h>


class EdgeDetection : public vole::Command {

	public:
		EdgeDetection();
		// use only with pre-filled config!
		EdgeDetection(const vole::EdgeDetectionConfig &config);
		int execute();

		void printShortHelp() const;
		void printHelp() const;

	protected:
		int executeSimple();

		vole::EdgeDetectionConfig config;

};


#endif
