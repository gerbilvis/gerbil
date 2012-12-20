/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>
	and Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef MANIPULATION_CONFIG_H
#define MANIPULATION_CONFIG_H

#include "vole_config.h"
#include <imginput.h>

namespace vole {

/**
 * Configuration parameters for the image cropping
 */
class ManipulationConfig : public Config {
public:

	ManipulationConfig(const std::string& prefix = std::string());

	/// input configuration
	ImgInputConfig m_InputConfig1;

	/// Optional secondary input (depends on task)
	ImgInputConfig m_InputConfig2;

	/// task
	std::string task;

	/// output filename
	std::string m_strOutputFilename;
	
	virtual std::string getString() const;

	virtual ~ManipulationConfig();

protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST
};

}

#endif // VOLE_CONFIG_H
