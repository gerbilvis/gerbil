#ifndef EDGE_DETECTION_CONFIG_H
#define EDGE_DETECTION_CONFIG_H

#include <vole_config.h>
#include <imginput_config.h>
#include <som_config.h>


namespace vole {

class EdgeDetectionConfig : public Config
{
public:
	EdgeDetectionConfig(const std::string& p = std::string());

	virtual ~EdgeDetectionConfig() {}

	// knn to use in SOM for edge image generation
	// int knn;

	// print out signed edgemap
	bool absolute;

	// output directory for edge_detection result
	std::string outputDir;

	// input is handled by imginput module
	ImgInputConfig imgInputCfg;

	SOMConfig somCfg;

	virtual std::string getString() const;

protected:

	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST
};

}

#endif
