#ifndef FELZENSZWALB_CONFIG_H
#define FELZENSZWALB_CONFIG_H

#include <vole_config.h>
#include <sm_config.h>
#include <imginput_config.h>

namespace seg_felzenszwalb {

class FelzenszwalbConfig : public Config {

public:
	FelzenszwalbConfig(const std::string& prefix = std::string());

	virtual ~FelzenszwalbConfig() {}

	// input is handled by imginput module
	imginput::ImgInputConfig input;
	/// output file name
	std::string output_file;

	float c;
	int min_size;
	bool eqhist;

	/// similarity measure for edge weighting
	similarity_measures::SMConfig similarity;

	virtual std::string getString() const;

protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST
};

}

#endif
