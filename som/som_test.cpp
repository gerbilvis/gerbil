/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "som_test.h"
#include "gensom.h"

#include <stopwatch.h>
#include <imginput.h>

//#include <command.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#ifdef WITH_BOOST
using namespace boost::program_options;
#endif

SOMTest::SOMTest()
 : vole::Command(
		"somtest",
		config,
		"Johannes Jordan",
		"johannes.jordan@cs.fau.de")
{}

SOMTest::~SOMTest() {}


int SOMTest::execute() {
	typedef boost::shared_ptr<GenSOM> GenSomPtr;
	GenSomPtr genSom;

	multi_img::ptr src = vole::ImgInput(config.imgInput).execute();
	if (src->empty())
		return 1;

	src->rebuildPixels(false);

	vole::Stopwatch watch	("Training");
	genSom = GenSomPtr(GenSOM::create(config.som, *src));

	cv::Mat3f bgr = genSom->bgr(src->meta, src->maxval);
	cv::imwrite(config.output_file, bgr * 255.f);
	return 0;
}


void SOMTest::printShortHelp() const {
	std::cout << "A simple test class for the SOM rewrite." << std::endl;
}


void SOMTest::printHelp() const {
	printShortHelp();
}



SOMTestConfig::SOMTestConfig(const std::string &p)
	: Config(p),
	  imgInput(prefix + "input"),
	  som(prefix + "som"),
	  output_file("som_cmf.png")
{
#ifdef WITH_BOOST
	initBoostOptions();
#endif // WITH_BOOST
}

std::string SOMTestConfig::getString() const
{
	return imgInput.getString() + som.getString();
}


#ifdef WITH_BOOST
void SOMTestConfig::initBoostOptions()
{
	options.add(imgInput.options);
	options.add(som.options);
	options.add_options()
			(key("output_file"), value(&output_file)->default_value(output_file),
			 "Filename for CMF representation of SOM.");
}
#endif // WITH_BOOST

