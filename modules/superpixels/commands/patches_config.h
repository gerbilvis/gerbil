#ifndef SUPERPIXELS_COMMANDS_PATCHESCONFIG
#define SUPERPIXELS_COMMANDS_PATCHESCONFIG

#include "vole_config.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace vole {

class PatchesConfig : public Config {
public:
	PatchesConfig( const std::string& prefix = std::string() );

public:
	virtual std::string getString() const;

protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif	// WITH_BOOST

public:
	int size_x;
	int size_y;
	int number_tries;
	std::string strategy;

private:
	friend class boost::serialization::access;
	// When the class Archive corresponds to an output archive, the
	// & operator is defined similar to <<.  Likewise, when the class Archive
	// is a type of input archive the & operator is defined similar to >>.
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & size_x;
		ar & size_y;
		ar & number_tries;
		ar & strategy;
	}
};

} // namespace vole

#endif // SUPERPIXELS_COMMANDS_PATCHESCONFIG
