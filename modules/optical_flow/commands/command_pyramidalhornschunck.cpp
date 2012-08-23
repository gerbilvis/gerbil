#include "command_pyramidalhornschunck.h"
#include "pyramidalhornschunckopticalflow.hpp"
#include "flow.hpp"
#include <opencv2/highgui/highgui.hpp>

namespace vole {

CommandPyramidalHornschunck::CommandPyramidalHornschunck():
	Command("pyramidal_hornschunck",
		config,
		"Sergiu Dotenco",
		"sergiu.dotenco@informatik.uni-erlangen.de"
	)
{
}

int CommandPyramidalHornschunck::execute()
{
  
  vision::PyramidalHornSchunckOpticalFlow<float> hs;
  hs.setAlpha(config.alpha);
  hs.setIterations(config.iterations);
  hs.setLevels(config.levels);
  hs.setScaleFactor(config.scale);

  cv::Mat_<float> u,v;

  cv::Mat1b previous = cv::imread(config.previous, CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat1b current = cv::imread(config.current, CV_LOAD_IMAGE_GRAYSCALE);

  hs.compute(previous, current, u, v);
  
  cv::Mat3b out(u.size());
  vision::middleburyDenseFlow(out, u, v);
  
  cv::imshow("denseflow",out);
  cv::imshow("current",current);
  cv::imshow("previous",previous);
  cv::waitKey(0);
  
  return 0;
}

void CommandPyramidalHornschunck::printShortHelp() const
{
	std::cout << "Dense optical flow after PyramidalHorn & Schunck." << std::endl;
}

void CommandPyramidalHornschunck::printHelp() const
{
	std::cout << "Dense optical flow after PyramidalHorn & Schunck." << std::endl;
}

} // vole
