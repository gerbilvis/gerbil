#include "command_similarity_measure.h"

#include "similarity_measure_core.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <sstream>

using namespace boost::program_options;

namespace vole {

// in the constructor we set up all parameters the user may configure. Note
// that the command name may deviate from the class name.
CommandSimilarityMeasure::CommandSimilarityMeasure()
 : vole::Command(
		"dist", // command name
		config,
		"Christian Riess", // author(s)
		"christian.riess@informatik.uni-erlangen.de") // email
{
}

int CommandSimilarityMeasure::execute() {
	bool exit_error = false;

	if (config.input_file1.length() < 1) { exit_error = true; std::cerr << "input1 must be set, aborted." << std::endl; }
	if (config.input_file2.length() < 1) { exit_error = true; std::cerr << "input2 must be set, aborted." << std::endl; }
	if (exit_error) return 1;
	cv::Mat_<unsigned char> image1 = cv::imread(config.input_file1, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat_<unsigned char> image2 = cv::imread(config.input_file2, CV_LOAD_IMAGE_GRAYSCALE);
	if (image1.cols < 1) { exit_error = true; std::cerr << "unable to read image1 (" << config.input_file1 << "), aborted." << std::endl; }
	if (image2.cols < 1) { exit_error = true; std::cerr << "unable to read image2 (" << config.input_file2 << "), aborted." << std::endl; }
	if (exit_error) return 1;
	if (!parseWindowDescription(config.win1, image1, config.window1)) {
		exit_error = true;
		std::cerr << "unable to parse description of win1 (" << config.win1
			<< "). Description must be four comma separated integer coordinates (x0, y0, x1, y1), aborted." << std::endl;
	}
	if (!parseWindowDescription(config.win2, image2, config.window2)) {
		exit_error = true;
		std::cerr << "unable to parse description of win2 (" << config.win2
			<< "). Description must be four comma separated integer coordinates (x0, y0, x1, y1), aborted." << std::endl;
	}
	if (exit_error) return 1;

	image1 = cv::Mat_<unsigned char>(image1, config.window1);
	image2 = cv::Mat_<unsigned char>(image2, config.window2);
	if ((image1.cols != image2.cols) || (image1.rows != image2.rows)) {
		exit_error = true;
		std::cerr << "both images (their subwindows, resp.) must have equal size, aborted." << std::endl;
	}
	if (!similarityMeasureCore.parseSelectedMetrics(config.selected_metrics, config.metrics)) { exit_error = true; }
	if (exit_error) return 1;

	similarityMeasureCore.setConfig(&config);
	similarityMeasureCore.setImages(image1, image2);
	std::vector<double> distances(config.metrics.size());
	similarityMeasureCore.getSimilarity(distances);
	for (unsigned int i = 0; i < distances.size(); ++i) {
		std::cout << "distance " << similarityMeasureCore.metricToString(config.metrics[i]) << ": " << distances[i] << std::endl;
	}
	

	return 0;
}

bool CommandSimilarityMeasure::parseWindowDescription(std::string windowDesc, cv::Mat_<unsigned char> &img, cv::Rect_<int> &window) {
	std::stringstream s;
	int x0, y0, x1, y1;
	std::string backup = windowDesc;
	if (windowDesc.length() < 1) {
		window.x = 0; window.y = 0;
		window.width = img.cols;
		window.height = img.rows;
		return true;
	}
	std::replace (windowDesc.begin(), windowDesc.end(), ',', ' '); 
	s << windowDesc;
	s >> x0 >> y0 >> x1 >> y1;
	if ((x0 >= x1) || (y0 >= y1)) {
		std::cerr << "in window description " << backup << ": x0 must be lower than x1, y0 lower than y1, aborted." << std::endl;
		return false;
	}
	if ((x0 < 0) || (y0 < 0)) {
		std::cerr << "in window description " << backup << ": x0, y0 must be greater or equal than 0, aborted." << std::endl;
		return false;
	}
	if ((x1 > img.cols) || (y1 > img.rows)) {
		std::cerr << "in window description " << backup << ": x1, y1 must be lower or equal to the image dimension, aborted." << std::endl;
		return false;
	}

	window.x = x0; window.y = y0;
	window.width = x1 - x0;
	window.height = y1 - y0;

	return true;
}

void CommandSimilarityMeasure::printShortHelp() const {
	std::cout << "\t2d grayscale img-to-img distance measures" << std::endl;
}


void CommandSimilarityMeasure::printHelp() const {
	std::cout << "2d grayscale img-to-img distance measures." << std::endl;
	std::cout << "computes the distance between two equally sized images." << std::endl;
	std::cout << "If the size of the images differs, coordinates of equally sized" << std::endl;
	std::cout << "subwindows must be provided." << std::endl;
}


} // namespace vole
