#ifndef OPTICALFLOW_COMMANDS_PYRAMIDALHORNSCHUNCKCONFIG
#define OPTICALFLOW_COMMANDS_PYRAMIDALHORNSCHUNCKCONFIG

#include "vole_config.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace vole {

class PyramidalHornschunckConfig : public Config {
public:
	PyramidalHornschunckConfig(const std::string& prefix = std::string());

public:
	virtual std::string getString() const;

protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST

public:
		std::string previous, current;
		float alpha;
		unsigned iterations;
		unsigned levels;
		float scale;


private:
	friend class boost::serialization::access;
	// When the class Archive corresponds to an output archive, the
	// & operator is defined similar to <<.  Likewise, when the class Archive
	// is a type of input archive the & operator is defined similar to >>.
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
	}
};

} // vole

#endif // OPTICALFLOW_COMMANDS_PYRAMIDALHORNSCHUNCKCONFIG
