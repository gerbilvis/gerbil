#ifndef SUPERPIXELS_COMMANDS_FELZENSZWALBCONFIG
#define SUPERPIXELS_COMMANDS_FELZENSZWALBCONFIG

#include "vole_config.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace vole {

class FelzenszwalbConfig : public Config {
public:
	FelzenszwalbConfig(const std::string& prefix = std::string());

public:
	virtual std::string getString() const;

protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST

public:
	std::string input_file;
	std::string input_file_16_bit;
	double max_intensity; // maximum intensity for a pixel
	double min_intensity; // minimum intensity for a pixel

	std::string output_file;

	bool deterministic_coloring;
	bool isGraphical;
	int kThreshold;
	double sigma;
	int min_size;

private:
	friend class boost::serialization::access;
	// When the class Archive corresponds to an output archive, the
	// & operator is defined similar to <<.  Likewise, when the class Archive
	// is a type of input archive the & operator is defined similar to >>.
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & input_file;
		ar & input_file_16_bit;
		ar & max_intensity; // maximum intensity for a pixel
		ar & min_intensity; // minimum intensity for a pixel
		ar & output_file;
		ar & kThreshold;
		ar & sigma;
		ar & min_size;
		ar & isGraphical;
	}
};

} // vole

#endif // SUPERPIXELS_COMMANDS_FELZENSZWALBCONFIG
