#ifndef OPTICALFLOW_COMMANDS_HORNSCHUNCKCONFIG
#define OPTICALFLOW_COMMANDS_HORNSCHUNCKCONFIG

#include "vole_config.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace vole {

class HornschunckConfig : public Config {
public:
	HornschunckConfig(const std::string& prefix = std::string());

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

#endif // OPTICALFLOW_COMMANDS_HORNSCHUNCKCONFIG
