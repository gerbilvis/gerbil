#include "meanshift_klresult.h"

#include <iostream>

namespace seg_meanshift {

void diagnoseKLResult(KLResult const& ret)
{
	if (ret.isState(KLState::Aborted)) {
		std::cerr << "findKL computation was aborted" << std::endl;
	} else if (ret.isState(KLState::NoneFound)) {
		std::cerr << "findKL computation found no solution" << std::endl;
	}
}

void KLResult::insertInto(std::map<std::string, boost::any> &dest)
{
	std::map<std::string, boost::any> rmap = makeKeyValueMap();
	dest.insert(rmap.begin(), rmap.end());
}

std::map<std::string, boost::any> KLResult::makeKeyValueMap() const
{
	std::map<std::string, boost::any> res;
	res["findKL.K"]       = K;
	res["findKL.L"]       = L;
	res["findKL.aborted"] = isState(KLState::Aborted);
	res["findKL.good"]    = isGood();
	return res;
}

} // namespace seg_meanshift
