#ifndef SUPERPIXELS_COMMANDS_VEKSLERCONFIG
#define SUPERPIXELS_COMMANDS_VEKSLERCONFIG

#include "vole_config.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace vole {

class VekslerConfig : public Config {
public:
	VekslerConfig(const std::string& prefix = std::string());

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
	// TODO: not sure if necessary, as passed to veksler::PlaceSeeds(..), veksler::MoveSeedsFromEdges(..), veksler::initializeLabeling(..)
	//       if necessary, maybe rename as done for Felzenszwalb? It's rather segments than patches and can be confusing with new PatchesSegmentation...
	int patchSize;
	int type;
	int iterations;
	int lambda;

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
		ar & patchSize;
		ar & type;
		ar & iterations;
		ar & lambda;
	}
};

} // vole

#endif // SUPERPIXELS_COMMANDS_VEKSLERCONFIG
