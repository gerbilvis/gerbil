#include "mfams.h"

#include <vector>
#include <algorithm>
#include <limits>

namespace seg_meanshift {

FAMS::MergedMode::MergedMode(const FAMS::Mode &d, int m, int spm)
	: members(m), spmembers(spm), data(d.data.size()), valid(true)
	{
		for (size_t i = 0; i < data.size(); ++i)
			data[i] = d.data[i];
}

std::vector<unsigned short> FAMS::MergedMode::normalized() const
{
	std::vector<unsigned short> ret(data.size());
	for (size_t i = 0; i < data.size(); ++i)
		ret[i] = data[i] / members;
	return ret;
}

double FAMS::MergedMode::distTo(const Mode &m) const
{
	double ret = 0.;
	for (size_t i = 0; i < data.size(); ++i)
		ret += std::abs(data[i] / members - m.data[i]);
	return ret;
}

void FAMS::MergedMode::add(const Mode &m, int sp)
{
	for (size_t i = 0; i < data.size(); ++i)
		data[i] += m.data[i];

	members++;
	spmembers += sp;
}

bool FAMS::MergedMode::invalidateIfSmall(int smallest)
{
	if (valid && members < smallest) {
		valid = false;
		return true;
	}
	return false;
}

std::pair<double, int>
FAMS::findClosest(const Mode& mode, const std::vector<MergedMode> &foomodes) {
	// distance and index
	std::pair<double, int> closest
			= std::make_pair(std::numeric_limits<double>::infinity(), -1);

	for (size_t i = 0; i < foomodes.size(); i++) {
		if (!foomodes[i].valid)
			continue;

		double dist = foomodes[i].distTo(mode);
		if (dist < closest.first) {
			closest.first = dist;
			closest.second = i;
		}
	}

	return closest;
}

void FAMS::trimModes(std::vector<MergedMode> &foomodes,
					 int npmin, bool sp, size_t allowance)
{
	// sort according to member count
	std::sort(foomodes.begin(), foomodes.end(), MergedMode::cmpSize);

	// find number of relevant modes (and keep at least one)
	size_t nrel = 1;
	for (; nrel < foomodes.size(); ++nrel) {
		if (npmin > (sp ? foomodes[nrel].spmembers : foomodes[nrel].members))
			break;
	}

	bgLog("ignoring %d modes smaller than %d points\n",
		  (foomodes.size() - nrel), npmin);
	if (nrel > allowance) {
		bgLog("exceeded allowance, only keeping %d modes\n",
			  allowance);
	}

	// shorten array accordingly
	foomodes.resize(std::min(nrel, allowance));
	// Note: invalidated modes are pruned by this, they have even less members
}

void FAMS::pruneModes()
{
	if (modes.empty())
		return;

	// use local copy of prune min. to be able to adapt it
	int npmin = config.pruneMinN;
	// compute jump		TODO: uses max. 10,000 points?
	int jm = (int)ceil(((double)modes.size()) / FAMS_PRUNE_MAXP);

	//** PASS ONE **//

	// set first mode
	std::vector<MergedMode> foomodes;
	foomodes.push_back(MergedMode(modes[0], 1,
					   (spsizes.empty() ? 1 : spsizes[0])));

	int invalid = 0; // for statistics on invalidated modes

	for (size_t cm = 1; cm < modes.size(); cm += jm) {

		/* compute closest mode */
		std::pair<double, int> closest = findClosest(modes[cm], foomodes);

		/* join */

		// good & cheap indicator for serious failure in DoFAMS()
		assert(modes[cm].window > 0);

		// closest mode is in range, so add point to it
		if (closest.first < (modes[cm].window >> FAMS_PRUNE_HDIV)) { // maybe *d_?
			int index = closest.second;

			// merge into mode
			foomodes[index].add(modes[cm], (spsizes.empty() ? 1 : spsizes[cm]));
		} else { // out of range, assume a new mode
			foomodes.push_back(MergedMode(modes[cm], 1,
									   (spsizes.empty() ? 1 : spsizes[cm])));
		}

		// when mode count gets overboard, invalidate modes with few members
		if (foomodes.size() > 2000) {
			for (size_t i = 0; i < foomodes.size(); ++i)
				invalid += (foomodes[i].invalidateIfSmall(3) ? 1 : 0);
		}
	}
	bgLog("done (%d modes left, %d of them have been invalidated)\n",
		  foomodes.size(), invalid);

	/* Trim modes */
	trimModes(foomodes, npmin, true, FAMS_PRUNE_MAXM);

	//** PASS TWO **//
	bgLog("            pass 2 ");

	/* Note: This code does not reset the mode information. Some pixels were
	 * added to the same modes before, some were added to modes that were cut
	 * off. In general only 10,000 pixels were considered so far (see
	 * FAMS_PRUNE_MAXP). So to do this properly, we would have to reset the
	 * counters, but also re-compute the mean vectors of the modes. Note: This
	 * is a problem of the original FAMS code, we only re-engineered the code.*/

	// HACK -- set npmin = 1 because we won't account to superpixel sizes
	if (!spsizes.empty())
		npmin = 1;

	for (size_t cm = 0; cm < modes.size(); ++cm) {

		/* compute closest mode */
		std::pair<double, int> closest = findClosest(modes[cm], foomodes);

		/* join -- this time don't care for window size */
		assert(closest.second >= 0);
		int index = closest.second;

		// merge into mode
		foomodes[index].add(modes[cm], (spsizes.empty() ? 1 : spsizes[cm]));
	}

	/* Trim modes, second time */
	trimModes(foomodes, npmin, false);

	/* store all relevant modes. */
	prunedModes.resize(foomodes.size());
	for (size_t i = 0; i < foomodes.size(); ++i) {
		prunedModes[i] = foomodes[i].normalized();
	}

	/* Now that we finally have a proper set of modes, last round to assign a
	 * mode index to each pixel. */
	prunedIndex.resize(modes.size());
	for (size_t cm = 0; cm < modes.size(); ++cm) {
		std::pair<double, int> closest = findClosest(modes[cm], foomodes);
		prunedIndex[cm] = closest.second;
	}

	bgLog("done pruning\n");
}

}
