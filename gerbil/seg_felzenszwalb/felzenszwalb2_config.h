#ifndef FELZENSZWALB_CONFIG_H
#define FELZENSZWALB_CONFIG_H

#include <vole_config.h>
#include <sm_config.h>
#include <multi_img.h>

namespace gerbil {

class FelzenszwalbConfig : public vole::Config {

public:
	FelzenszwalbConfig(const std::string& prefix = std::string());

	virtual ~FelzenszwalbConfig() {}

	/// input file name
	std::string input_file;
	/// output file name
	std::string output_file;

	float c;
	int min_size;
	bool eqhist;

	/// similarity measure for edge weighting
	vole::SMConfig similarity;

	virtual std::string getString() const;

protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST
};

}

#endif
