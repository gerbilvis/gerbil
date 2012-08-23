/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_ERROR_H
#define VOLE_ERROR_H

class VoleError {
public:
	static void warn(std::string warning, int linenumber = -1, std::string filename = "") {
		std::cerr << "Warning: ";
		output(warning, linenumber, filename);
	}

	static void err(std::string error, int linenumber = -1, std::string filename = "") {
		std::cerr << "Error: ";
		output(error, linenumber, filename);
	}

protected:
	static void output(std::string error, int linenumber = -1, std::string filename = "") {
		if (linenumber != -1) {
			std::cerr << "Line " << linenumber;
			if (filename != "") {
				std::cerr << " in " << filename << ": ";
			} else {
				std::cerr << ": ";
			}
		} else {
			if (filename != "") {
				std::cerr << "In " << filename << ": ";
			}
		}
		std::cerr << error << std::endl;
	}
};

#endif // VOLE_ERROR_H

