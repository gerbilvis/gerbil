#include "som_config.h"
#include <time.h>

#ifdef WITH_BOOST_PROGRAM_OPTIONS
using namespace boost::program_options;
#endif

namespace som {

ENUM_MAGIC(som, Type)

SOMConfig::SOMConfig(const std::string& p)
	: Config(p),
	  type(SOM_SQUARE),
	  dsize(32),
	  seed(time(NULL)),
	  maxIter(40000),
	  learnStart(0.75),
	  learnEnd(0.01), // TODO: we stop updating, when weight is < 0.01!
	  sigmaStart(12.), // ratio sigmaStart : sigmaEnd should be about 4 : 1
	  sigmaEnd(2.),
	  gaussKernel(false),
//    use_opencl(false),
//    use_opencl_cpu_opt(false),
	  somFile(),
	  similarity(prefix + "similarity")
{
	#ifdef WITH_BOOST_PROGRAM_OPTIONS
		initBoostOptions();
	#endif // WITH_BOOST
}

// descriptions of configuration options
namespace desc {
DESC_OPT(type,
		"Layout of the neurons in the SOM: square, cube, or tesseract")
DESC_OPT(dsize,
		"Number of neurons per dimension (for isometric SOMs)")
DESC_OPT(maxIter,
		"Number of training iterations for the SOM")
DESC_OPT(learnStart,
		"Learning rate at the beginning")
DESC_OPT(learnEnd,
		"Learning rate at the end of the training process")
DESC_OPT(sigmaStart,
		"Initial neighborhood radius")
DESC_OPT(sigmaEnd,
		"Neighborhood radius at the end of the training process")
DESC_OPT(seed,
		"Seed value of random number generators")
DESC_OPT(gaussKernel,
		"Use gaussian kernel instead of uniform kernel")
DESC_OPT(use_opencl,
		"Use OpenCL to accelerate computations")
DESC_OPT(use_opencl_cpu_opt,
		"Use OpenCL to accelerate computations (CPU optimized version)")
DESC_OPT(somFile,
		"If file exists read binary SOM format, "
		"otherwise write after training. If not set (empty string), do neither.")
}

#ifdef WITH_BOOST_PROGRAM_OPTIONS
void SOMConfig::initBoostOptions() {
	options.add_options()
		BOOST_OPT(type)
		BOOST_OPT(dsize)
		BOOST_OPT(maxIter)
		BOOST_OPT(learnStart)
		BOOST_OPT(learnEnd)
		BOOST_OPT(sigmaStart)
		BOOST_OPT(sigmaEnd)
		BOOST_OPT(seed)
		BOOST_BOOL(gaussKernel)
		//BOOST_BOOL(use_opencl)
		//BOOST_BOOL(use_opencl_cpu_opt)
		BOOST_OPT(somFile)
		;
	options.add(similarity.options);

	if (prefix_enabled)	// skip input/output options
		return;
}
#endif // WITH_BOOST

std::string SOMConfig::getString() const {
	std::stringstream s;
	if (prefix_enabled)
		s << "[" << prefix << "]" << std::endl;

	COMMENT_OPT(s, type);
	COMMENT_OPT(s, dsize);
	COMMENT_OPT(s, maxIter);
	COMMENT_OPT(s, learnStart);
	COMMENT_OPT(s, learnEnd);
	COMMENT_OPT(s, sigmaStart);
	COMMENT_OPT(s, sigmaEnd);
	COMMENT_OPT(s, seed);
	COMMENT_OPT(s, gaussKernel);
	s  << similarity.getString();
	return s.str();
}

}
