#include "RBC_config.h"

#ifdef WITH_BOOST
using namespace boost::program_options;
#endif

namespace vole {

RBCConfig::RBCConfig(const std::string &prefix)
    : Config(prefix)
{
#ifdef WITH_BOOST
    initBoostOptions();
#endif // WITH_BOOST
}

#ifdef WITH_BOOST
void RBCConfig::initBoostOptions()
{
    options.add_options()
        (key("input,I"), value(&input_file)->default_value("input.png"),
         "Image file to process")
        (key("gradient,G"), value(&gradient)->default_value(false),
         "Compute gradient")
        (key("output,O"), value(&output_dir)->default_value("/tmp/"),
         "Output directory")
        (key("maxQuery,M"), value(&maxQuerySize)->default_value(1024),
         "Maximum query size performed on RBC structure")
        (key("numReps,R"), value(&numReps)->default_value(0),
         "Number of representatives randomly selected from original "
         "dataset to create RBC structure. Value 0 means value equal "
         "to sqrt(total_num_of_points)*5.")
        (key("pointsPerRepr,M"), value(&pointsPerRepr)->default_value(16*1024),
         "Number of points assigned to every representative in RBC structure.")
        (key("pilotsThreshold,T"), value(&pilotsThreshold)->default_value(512),
         "Number of points used to calculate pilot value")

        (key("old,D"), value(&old_impl)->default_value(false),
         "Invocation of original main function");
}
#endif

std::string RBCConfig::getString() const
{
    return "parameters description, to do\n";
}


}
