/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef TALKINGWATCH_H
#define TALKINGWATCH_H

#ifdef WITH_OPENCV2 // theoretically, vole could be built w/o opencv..

#include <cv.h>
#include <string>
#include <sstream>
#include <iostream>

namespace vole {

/** Class for easy CPU time measurement using OpenCV functions.

Use this class to easily track the running time of your algorithm. It will
start on creation and stop+report on destruction. Usage example:
\code
	{
		Talkingwatch s("My awesome algorithm");

		// my awesome algorithm here
	} // will report here and return execution time
\endcode
Another usage example:
\code
	Stopwatch s; // no string given, no reporting on destruction

	// do some stuff

	s.print("some stuff"); // trigger report manually
	s.reset(); // reset the clock

	// do some other stuff

	s.print("some other stuff");

	// ...
\endcode

**/
struct Talkingwatch {
	Talkingwatch(std::string name = std::string())
	 : name(name), verbose(!name.empty()) { reset(); }

	~Talkingwatch() { if (verbose)	print(name); }

	/** Reset the clock for new measurement. */
	void reset() { time = cv::getTickCount(); }

	/** Return elapsed time in seconds. */
	double measure() const
	{ return ((double)(cv::getTickCount() - time))/cv::getTickFrequency(); }

	/** Report to cerr */
	std::string print(const std::string &descript) const
	{
		std::stringstream out;
		std::ios_base::fmtflags orig_flags = std::cerr.flags();
		std::streamsize orig_precision = std::cerr.precision();
		std::cerr.setf(std::ios_base::fixed);
		std::cerr.precision(6);
		std::cerr.width(10);
		out << measure() << " s: " << descript << std::endl;
		out.flags(orig_flags); std::cerr.precision(orig_precision);
		
		return out.str();
	}

	/** Report to cerr, then reset the clock for new measurement. */
	void print_reset(const std::string &descript) {
		print(descript);
		reset();
	}

	std::string name;
	bool verbose;
	int64 time;
};

}

#endif // WITH_OPENCV2
#endif // TALKINGWATCH_H
