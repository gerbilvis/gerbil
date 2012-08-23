#include "example_connected_components.h"

#include "connected_components.h"

#include "cv.h"
#include "highgui.h"

using namespace boost::program_options;

namespace vole {
	namespace examples {

ConnectedComponents::ConnectedComponents()
 : vole::Command(
		"example_connected_components", // command name
		config,
		"Christian Riess", // author(s)
		"christian.riess@informatik.uni-erlangen.de") // email
{
}

// the execute method is the starting point for every vole command
int ConnectedComponents::execute() {

	cv::Mat_<int> cc_input;
	int threshold = 1;
	// first of all we should check if there is an input file given - if not,
	// we create a synthetic example
	if (config.input_file.length() < 1) {
		std::cout << "ERROR: No input file name given, creating synthetic example" << std::endl << std::endl;
		
		// let's create a cross example - the four corners are below or equal to our threshold.

		cc_input = cv::Mat_<int>(9, 9, static_cast<int>(1));
		for (int y = 0; y < 3; y++) {
			for (int x = 3; x < 6; x++) {
				cc_input[y  ][x  ] = 2;
				cc_input[y+3][x-3] = 2;
				cc_input[y+3][x  ] = 2;
				cc_input[y+3][x+3] = 2;
				cc_input[y+6][x  ] = 2;
			}
		}
	} else {
		// do some thresholding over the file
		cv::Mat_<cv::Vec3b> img = cv::imread(config.input_file);
		// find max val;
		unsigned char maxVal = 0;
		for (int y = 0; y < img.rows; ++y) {
			for (int x = 0; x < img.cols; ++x) {
				for (int c = 0; c < 3; ++c)
					if (img[y][x][3] > maxVal)
						maxVal = img[y][x][c];
			}
		}
		// heuristic example: deselect every point where the red color channel is smaller than 2*maxVal/3
		double threshold = 2.0*maxVal / 3.0;
		cc_input = cv::Mat_<int>(img.rows, img.cols, static_cast<int>(1));
		for (int y = 0; y < img.rows; ++y) {
			for (int x = 0; x < img.cols; ++x) {
				if (img[y][x][0] < threshold) { cc_input[y][x] = 2; }
			}
		}
	}

	vole::ConnectedComponents cc(1, cc_input);
	cc.computeConnectedComponents();
	std::cout << " there are " << cc.getNumberOfConnectedComponents() << " components " << std::endl;

return 0;
}


void ConnectedComponents::printShortHelp() const {
	std::cout << "\tExample for connected components computation" << std::endl;
}


void ConnectedComponents::printHelp() const {
	std::cout << "Example for the connected components computation." << std::endl;
}


	} // namespace examples
} // namespace vole
