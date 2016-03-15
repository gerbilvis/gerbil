#include "mapper_config.h"

namespace mapper {

// descriptions of configuration options
namespace desc {
DESC_OPT(mask, "Mask file (black means excluded, else included)")
DESC_OPT(output, "Output image file")
}

MapperConfig::MapperConfig(const std::string &p)
    : Config(p),
      mask("mask.png"),
      output("output.png"),
      input(prefix + "input"),
      similarity(prefix + "similarity")
{
	options.add_options()
	        BOOST_OPT(mask)
	        BOOST_OPT(output)
	        ;
	options.add(input.options);
	options.add(similarity.options);
}

}
