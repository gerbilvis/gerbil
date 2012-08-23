#include "patches_config.h"

#ifdef WITH_BOOST_PROGRAM_OPTIONS
using namespace boost::program_options;
#endif // WITH_BOOST_PROGRAM_OPTIONS

namespace vole {

PatchesConfig::PatchesConfig( const std::string& prefix )
	: Config( prefix )
{
	// TODO: we don't want this next line, we want derived verbosity values for all configs derived from vole::Config:
	verbosity = 1;

	#ifdef WITH_BOOST_PROGRAM_OPTIONS
		initBoostOptions();
	#endif // WITH_BOOST_PROGRAM_OPTIONS
}

std::string PatchesConfig::getString() const
{
	std::stringstream s;

	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	}

	s << "size_x=" << size_x << " # patchsize in x-direction, will be swapped randomly with size_y." << std::endl;
	s << "size_y=" << size_y << " # patchsize in y-direction, will be swapped randomly with size_x." << std::endl;
	s << "number_tries=" << number_tries << " # number of patches to try for the voting" << std::endl;
	s << "strategy=" << strategy << " # selection strategy if a patch is crossing a segment's borders. {random, largest, all}" << std::endl;

	return s.str();
}

#ifdef WITH_BOOST_PROGRAM_OPTIONS
void PatchesConfig::initBoostOptions()
{
	options.add_options()( key( "size_x" ), value( &size_x )->default_value( 40 ), "patchsize in x-direction, will be swapped randomly with size_y." );
	options.add_options()( key( "size_y" ), value( &size_y )->default_value( 10 ), "patchsize in y-direction, will be swapped randomly with size_x." );
	options.add_options()( key( "number_tries" ), value( &number_tries )->default_value( 1000 ), "number of patches to try for the voting" );
	options.add_options()( key( "strategy" ), value( &strategy )->default_value( "largest" ), "selection strategy if a patch is crossing a segment's borders. {random, largest, all}" );
}
#endif // WITH_BOOST_PROGRAM_OPTIONS

} // namespace vole
