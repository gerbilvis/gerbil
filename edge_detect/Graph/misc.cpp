//	-------------------------------------------------------------------------------------------------------------	 	//
// 														Variants of Self-Organizing Maps																											//
// Studienarbeit in Computer Vision at the Chair of Patter Recognition Friedrich-Alexander Universitaet Erlangen		//
// Start:	15.11.2010																																																//
// End	:	16.05.2011																																																//
// 																																																									//
// Ralph Muessig																																																		//
// ralph.muessig@e-technik.stud.uni-erlangen.de																																			//
// Informations- und Kommunikationstechnik																																					//
//	---------------------------------------------------------------------------------------------------------------	//


#include "misc.h"
#include <iostream>

Color::Color(unsigned int r, unsigned int g, unsigned int b) : red(r), green(g), blue(b)
{
	if(r > 255)red = 255;
	if(g > 255)green = 255;
	if(b > 255)blue = 255;  
}

Color::Color() : red((unsigned int)0), green((unsigned int)0), blue((unsigned int)0)
{}

Color::Color(unsigned int c) : red(c), green(c), blue(c)
{
	if(c > 255)
		c = 255;
	red = green = blue = c;
}

Color::~Color(){}

std::string Color::rgb2hex()
{
	std::stringstream ss;
	if(red < 16)ss << "0";
	ss <<std::hex << (int)red ;
	if(green < 16)ss << "0";
	ss <<std::hex << (int)green ;
	if(blue < 16)ss << "0";
	ss <<std::hex << (int)blue ;

	std::string ret(ss.str());
	
	return ret;
}

std::string Color::rgb2string()
{
	std::stringstream ss;
	ss << red << " " << green << " " << blue <<std::endl;
	std::string ret(ss.str());
	return ret;
}
