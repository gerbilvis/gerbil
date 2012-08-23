/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_STORAGE_H
#define VOLE_STORAGE_H

#include "cv.h"
#include <sstream>

namespace vole {

class Storage {
public:
	static void write(const char *filename, CvMat *blob, const char *description = "no_desc");
	static void write(const char *filename, cv::Mat blob, const char *description = "no_desc");

	static cv::Mat read(const char *filename, const char *description = "no_desc");

	static std::string getStorageLocation(std::string filename, std::string suffix) {
		std::stringstream s;
		s << "/tmp/" << getDescription(filename, suffix);
		return s.str();
	}

	static std::string getDescription(std::string filename, std::string suffix) {
		std::stringstream s;
		s << filename << "_" << suffix << "_storage.yml";
		std::string result = s.str();
		for (unsigned int i = 0; i < result.length(); ++i) if (result[i] == '/') result[i] = '_';
		return result;
	}

};

}

#endif // VOLE_STORAGE_H
