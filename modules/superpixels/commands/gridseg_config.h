#ifndef SUPERPIXELS_COMMANDS_GRIDSEG_CONFIG
#define SUPERPIXELS_COMMANDS_GRIDSEG_CONFIG

#include "vole_config.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace vole {

class GridSegConfig : public Config {
public:
	GridSegConfig(const std::string& prefix = std::string());

public:
	virtual std::string getString() const;

protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST

public:
	std::string output_file;

	int x_dim;
	int y_dim;
//	bool deterministic_coloring;
	int block_size;
	int number_blocks;
	std::string prior_segmentation;
	bool fuse_max_area;

private:
	friend class boost::serialization::access;
	// When the class Archive corresponds to an output archive, the
	// & operator is defined similar to <<.  Likewise, when the class Archive
	// is a type of input archive the & operator is defined similar to >>.
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & x_dim;
		ar & y_dim;
		ar & output_file;
		ar & block_size;
		ar & number_blocks;
		ar & prior_segmentation;
	}
};

} // vole

#endif // SUPERPIXELS_COMMANDS_GRIDSEG_CONFIG
