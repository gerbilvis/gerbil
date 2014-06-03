#ifndef GRAPHSEG_CONFIG_H
#define GRAPHSEG_CONFIG_H

#include <vole_config.h>
#include <sm_config.h>
#include <imginput_config.h>
#ifdef WITH_SOM
#include <som_config.h>
#endif
#include <multi_img.h>

namespace seg_graphs {

enum algorithm {
	KRUSKAL,
	PRIM,
	WATERSHED2
};
#define seg_graphs_algorithmString {"KRUSKAL", "PRIM", "WATERSHED2"}

/**
 * Configuration parameters for the graph cut / power watershed segmentation
 */
class GraphSegConfig : public Config {

public:
	GraphSegConfig(const std::string& prefix = std::string());

	virtual ~GraphSegConfig() {}

	// input is handled by imginput module
	imginput::ImgInputConfig input;
	/// seed input filename
	std::string seed_file;
	/// output file name
	std::string output_file;

	/// seed file contains fore/background labels (false) or >2 labels (true)
	bool multi_seed;
	/// algorithm to be employed
	algorithm algo;
	/// use geodesic reconstruction of the weights
	bool geodesic;

	/// similarity measure for edge weighting
	similarity_measures::SMConfig similarity;

#ifdef WITH_SOM
	/// use SOM similarity instead
	bool som_similarity;
	som::SOMConfig som;
#endif

	virtual std::string getString() const;

protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST
};

}

#endif
