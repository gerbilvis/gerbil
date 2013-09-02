#ifndef VOLE_COMMAND_SIMILARITY_MEASURE_H
#define VOLE_COMMAND_SIMILARITY_MEASURE_H

#include <iostream>
#include <string>
#include "command.h"
#include "similarity_measure_config.h"
#include "similarity_measure_core.h"

namespace vole {

// our class starts here
class CommandSimilarityMeasure : public vole::Command {
public:
	CommandSimilarityMeasure();

	int execute();

	void printShortHelp() const;
	void printHelp() const;


private:

	bool parseWindowDescription(std::string windowDesc, cv::Mat_<unsigned char> &img, cv::Rect_<int> &window);

	SimilarityMeasureConfig config;
	SimilarityMeasureCore similarityMeasureCore;
};

}

#endif // VOLE_COMMAND_SIMILARITY_MEASURE_H
