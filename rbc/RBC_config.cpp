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
        (key("old,D"), value(&old_impl)->default_value(false),
         "Invocation of original main function");
}
#endif

std::string RBCConfig::getString() const
{
    return "parameters description, to do\n";
}


}
