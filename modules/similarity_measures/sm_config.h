/*	
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef FACTORY_H
#define FACTORY_H

#include <vole_config.h>
#include <multi_img.h>

/**
	Plugin config for other methods that rely on similarity measures.
	Include this as a member to your config class.

	Use the method SMFactory::spawn() to create the similarity
	measurement function chosen by the user.
**/

namespace vole {

enum similarity_fun {
	MANHATTAN,
	EUCLIDEAN,
	CHEBYSHEV,
	MOD_SPEC_ANGLE,
	SPEC_INF_DIV,
	SIDSAM1,
	SIDSAM2,
	NORM_L2
};
#define similarity_funString {"MANHATTAN", "EUCLIDEAN", "CHEBYSHEV", \
	"MOD_SPEC_ANGLE", "SPEC_INF_DIV", "SIDSAM1", "SIDSAM2", "NORM_L2"}

class SMConfig: public Config {
public:

	SMConfig(const std::string& prefix = std::string());

	// similarity measure selection
	similarity_fun measure;

	virtual std::string getString() const;

	virtual ~SMConfig() {}

protected:
#ifdef WITH_BOOST
	virtual void initBoostOptions();
#endif // WITH_BOOST
};

}

#endif
