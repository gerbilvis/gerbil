#include "example_graham_scan.h"

#include "graham_scan.h"

#include "cv.h"
#include "highgui.h"

using namespace boost::program_options;

namespace vole {
	namespace examples {

// in the constructor we set up all parameters the user may configure. Note
// that the command name may deviate from the class name.
GrahamScan::GrahamScan()
 : vole::Command(
		"example_graham_scan", // command name
		config,
		"Christian Riess", // author(s)
		"christian.riess@informatik.uni-erlangen.de") // email
{
}

// the execute method is the starting point for every vole command
int GrahamScan::execute() {

	std::vector<cv::Vec2f> input_points;
	std::vector<cv::Vec2f> hull;

	// first of all we should check if there is an input file given - if not,
	// we create a synthetic example
	if (config.input_file.length() < 1) {
		std::cout << "ERROR: No input file name given, creating synthetic example" << std::endl << std::endl;
		// and quit - the main method will print our available options if we return 1
		
		input_points.push_back(cv::Vec2f(45, 3));
		input_points.push_back(cv::Vec2f(46, 3));
		input_points.push_back(cv::Vec2f(47, 3));
		input_points.push_back(cv::Vec2f(45, 4));
		input_points.push_back(cv::Vec2f(46, 4));
		input_points.push_back(cv::Vec2f(47, 4));
		input_points.push_back(cv::Vec2f(45, 5));
		input_points.push_back(cv::Vec2f(46, 5));
		input_points.push_back(cv::Vec2f(47, 5));
		input_points.push_back(cv::Vec2f(34, 6));
		input_points.push_back(cv::Vec2f(39, 6));
		input_points.push_back(cv::Vec2f(40, 6));
		input_points.push_back(cv::Vec2f(41, 6));
		input_points.push_back(cv::Vec2f(42, 6));
		input_points.push_back(cv::Vec2f(43, 6));
		input_points.push_back(cv::Vec2f(44, 6));
		input_points.push_back(cv::Vec2f(45, 6));
		input_points.push_back(cv::Vec2f(46, 6));
		input_points.push_back(cv::Vec2f(47, 6));
		input_points.push_back(cv::Vec2f(34, 7));
		input_points.push_back(cv::Vec2f(35, 7));
		input_points.push_back(cv::Vec2f(36, 7));
		input_points.push_back(cv::Vec2f(37, 7));
		input_points.push_back(cv::Vec2f(38, 7));
		input_points.push_back(cv::Vec2f(39, 7));
		input_points.push_back(cv::Vec2f(40, 7));
		input_points.push_back(cv::Vec2f(41, 7));
		input_points.push_back(cv::Vec2f(42, 7));
		input_points.push_back(cv::Vec2f(43, 7));
		input_points.push_back(cv::Vec2f(44, 7));
		input_points.push_back(cv::Vec2f(45, 7));
		input_points.push_back(cv::Vec2f(46, 7));
		input_points.push_back(cv::Vec2f(47, 7));
		input_points.push_back(cv::Vec2f(48, 7));
		input_points.push_back(cv::Vec2f(34, 8));
		input_points.push_back(cv::Vec2f(45, 8));
		input_points.push_back(cv::Vec2f(34, 9));
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
		// heuristic example: pick every point where 2 color channels are larger than 2*maxVal/3
		double threshold = 2.0*maxVal / 3.0;
		for (int y = 0; y < img.rows; ++y) {
			for (int x = 0; x < img.cols; ++x) {
				int doit = 0;
				for (int c = 0; c < 3; ++c) { if (img[y][x][c] > threshold) doit++; }
				if (doit >= 2) input_points.push_back(cv::Vec2f(static_cast<float>(x), static_cast<float>(y)));
			}
		}
	}
	vole::GrahamScan gs;
	gs.getHull(input_points, hull);
	std::cout << "there are " << hull.size() << " points on the hull: " << std::endl;
	for (unsigned int i = 0; i < hull.size(); ++i)
		std::cout << "(" << hull[i][0] << ", " << hull[i][1] << ")" << std::endl;

return 0;
}


void GrahamScan::printShortHelp() const {
	std::cout << "\tExample for the Graham Scan 2D convex hull computation" << std::endl;
}


void GrahamScan::printHelp() const {
	std::cout << "Example for the Graham Scan 2D convex hull computation." << std::endl;
	std::cout << "If an input image is given, the convex hull is computed on a number of points that have" << std::endl;
	std::cout << "in two color channels intensities that are higher than (2/3)*maxIntensity." << std::endl;
	std::cout << "If no input image is given, a synthetic example is created. This consists of a cross. The" << std::endl;
	std::cout << "corresponding convex hull is an eight-edge." << std::endl;
}


	} // namespace examples
} // namespace vole
