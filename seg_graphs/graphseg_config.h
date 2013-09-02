#ifndef GRAPHSEG_CONFIG_H
#define GRAPHSEG_CONFIG_H

#include <vole_config.h>
#include <sm_config.h>
#ifdef WITH_EDGE_DETECT
#include <edge_detection_config.h>
#endif
#include <multi_img.h>

namespace vole {

enum graphsegalg {
	KRUSKAL,
	PRIM,
	WATERSHED2
};
#define graphsegalgString {"KRUSKAL", "PRIM", "WATERSHED2"}

/**
 * Configuration parameters for the graph cut / power watershed segmentation
 */
class GraphSegConfig : public Config {

public:
	GraphSegConfig(const std::string& prefix = std::string());

	virtual ~GraphSegConfig() {}

	/// input file name
	std::string input_file;
	/// seed input filename
	std::string seed_file;
	/// output file name
	std::string output_file;

	/// seed file contains fore/background labels (false) or >2 labels (true)
	bool multi_seed;
	/// algorithm to be employed
	graphsegalg algo;
	/// use geodesic reconstruction of the weights
	bool geodesic;

	/// similarity measure for edge weighting
	SMConfig similarity;

#ifdef WITH_EDGE_DETECT
	/// use SOM similarity instead
	bool som_similarity;
	EdgeDetectionConfig som;
#endif

	virtual std::string getString() const;

protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST
};

}

#endif
