#include "example_minimum_enclosing_rectangle.h"
#include "minimum_enclosing_rectangle.h"

#include "cv.h"
#include "highgui.h"

#include <vector>

using namespace boost::program_options;

namespace vole {
	namespace examples {

// in the constructor we set up all parameters the user may configure. Note
// that the command name may deviate from the class name.
MinimumEnclosingRectangle::MinimumEnclosingRectangle()
 : vole::Command(
		"example_mer", // command name
		config,
		"Christian Riess", // author(s)
		"christian.riess@informatik.uni-erlangen.de") // email
{
}

// the execute method is the starting point for every vole command
int MinimumEnclosingRectangle::execute() {

	std::vector<cv::Vec2f> hull_points;
	std::vector<cv::Vec2f> mer_points;

	// first of all we should check if there is an input file given - if not,
	// we immediately abort
	if (config.input_file.length() < 1) {
		std::cout << "ERROR: No input file name given, creating a synthetic example." << std::endl << std::endl;

		hull_points.push_back( cv::Vec2f (19, 4) );
		hull_points.push_back( cv::Vec2f (31, 7) );
		hull_points.push_back( cv::Vec2f (31, 8) );
		hull_points.push_back( cv::Vec2f (30, 9) );
		hull_points.push_back( cv::Vec2f (22, 13));
		hull_points.push_back( cv::Vec2f (20, 13));
		hull_points.push_back( cv::Vec2f (16, 11));
		hull_points.push_back( cv::Vec2f (11, 8) );


		for (unsigned int i = 0; i < hull_points.size(); ++i)
			std::cout << "hull point " << i << ": (" << hull_points[i][0] << ", " << hull_points[i][1] << ")" << std::endl;

		vole::MinimumEnclosingRectangle mer;
		mer.setPoints(hull_points);

		mer.getMerOSquared(mer_points);

		for (unsigned int i = 0; i < mer_points.size(); ++i)
			std::cout << "mer  point " << i << ": (" << mer_points[i][0] << ", " << mer_points[i][1] << ")" << std::endl;

		// and quit - the main method will print our available options if we return 1
		return 0;
	} else {
		std::cout << "oops, input file given, but nothing is implemented here :((" << std::endl;
	}


	return 0;
}


void MinimumEnclosingRectangle::printShortHelp() const {
	std::cout << "\tExample application for the minimum enclosing rectangle computation (\"rotating calipers\")" << std::endl;
}


void MinimumEnclosingRectangle::printHelp() const {
	std::cout << "TODO:	" << std::endl;
}

}
}
