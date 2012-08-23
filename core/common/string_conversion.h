#ifndef STRING_CONVERSION_H
#define STRING_CONVERSION_H

#include <sstream>
#include <vector>

namespace vole {

class StringConversion {
public:
	// FIXME no error checking
	static double toDbl(std::string str);
	static float toFlt(std::string str);
	static int toInt(std::string str);
	static unsigned int toUInt(std::string str);
	static std::vector<std::string> stringToList(std::string str, char split_character);
	static std::vector<int> stringToIntList(std::string str, char split_character);

#ifdef WITH_BOOST_PROGRAM_OPTIONS
	static std::vector<std::string> tokenizeLine(const std::string& line, char tok);
#endif

};

}

#endif // STRING_CONVERSION_H
