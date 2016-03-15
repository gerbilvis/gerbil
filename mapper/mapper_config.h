#ifndef MAPPER_CONFIG_H
#define MAPPER_CONFIG_H

#include <vole_config.h>
#include <imginput_config.h>
#include <sm_config.h>

namespace mapper {

struct MapperConfig : public Config
{
	MapperConfig(const std::string& prefix = std::string());

	std::string mask;
	std::string output;

	imginput::ImgInputConfig input;
	similarity_measures::SMConfig similarity;
};

}

#endif // MAPPER_CONFIG_H
