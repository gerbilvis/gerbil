#include "solution.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip> 
 
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
  if(c > 255)c = 255;
  red = green = blue = c;
}

Color::~Color(){}

std::string Color::rgb2hex()
{
//   sprintf(buffer, "%2X", red);
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

 
Solution::Solution(int const *p, int o) :  obj(o)
{
//   *pos = p;
}

Solution::Solution(int * const *p, const int &o)
{

}

Solution::Solution(int object) :obj(object)
{}

Solution::Solution(int object,Color color) :obj(object)
{

  c.red = color.red;
  c.green = color.green;
  c.blue = color.blue;

}

Color Solution::getColor() 
{
  return c;
}

void Solution::setColor(Color color)
{
  c = color;
}

Solution::Solution()
{}

Solution::~Solution()
{}
 
 