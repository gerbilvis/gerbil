/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "string_conversion.h"
#include <sstream>

#ifdef WITH_BOOST_PROGRAM_OPTIONS
#include <boost/tokenizer.hpp>
#endif // WITH_BOOST_PROGRAM_OPTIONS

namespace vole {

	double StringConversion::toDbl(std::string str) {
		std::stringstream s;
		s << str;
		double res;
		s >> res;
		return res;
	}

	float StringConversion::toFlt(std::string str) {
		std::stringstream s;
		s << str;
		float res;
		s >> res;
		return res;
	}

	int StringConversion::toInt(std::string str) {
		std::stringstream s;
		s << str;
		int res;
		s >> res;
		return res;
	}

	unsigned int StringConversion::toUInt(std::string str) {
		std::stringstream s;
		s << str;
		unsigned int res;
		s >> res;
		return res;
	}

	std::vector<std::string> StringConversion::stringToList(std::string str, char split_character) {
		std::vector<std::string> splitted;
		unsigned int last_i = 0;
		int i;
		for (i = 0; i < static_cast<int>(str.length()); ++i) {
			if (str[i] == split_character) {
				if (static_cast<int>(last_i) < i) splitted.push_back(str.substr(last_i, i - last_i));
				last_i = i+1;
			}
		}
		if (static_cast<int>(last_i) < i) splitted.push_back(str.substr(last_i, i - last_i));
		return splitted;
	}

	std::vector<int> StringConversion::stringToIntList(std::string str, char split_character) {
		std::vector<int> splitted;
		unsigned int last_i = 0;
		int i;
		for (i = 0; i < static_cast<int>(str.length()); ++i) {
			if (str[i] == split_character) {
				if (static_cast<int>(last_i) < i) {
					std::stringstream s;
					s << str.substr(last_i, i - last_i);
					int tmp;
					s >> tmp;
					splitted.push_back(tmp);
				}
				last_i = i+1;
			}
		}
		if (static_cast<int>(last_i) < i) {
			std::stringstream s;
			s << str.substr(last_i, i - last_i);
			int tmp;
			s >> tmp;
			splitted.push_back(tmp);
		}
		return splitted;
	}

#ifdef WITH_BOOST_PROGRAM_OPTIONS
	std::vector<std::string> StringConversion::tokenizeLine(const std::string& line, char tok)
	{
		std::vector<std::string> tokens;

		boost::tokenizer<boost::escaped_list_separator<char> > tk(line, boost::escaped_list_separator<char>('\\', tok, '\"'));

		for (boost::tokenizer<boost::escaped_list_separator<char> >::iterator it(tk.begin()); it != tk.end(); it++) {
			tokens.push_back(*it);
		}

		return tokens;
	}
#endif // WITH_BOOST_PROGRAM_OPTIONS

}
