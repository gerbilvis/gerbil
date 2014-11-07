/////////////////////////////////////////////////////////////////////////////
// Name:        fams.cpp
// Purpose:     fast adaptive meanshift implementation
// Author:      Ilan Shimshoni
// Modified by: Bogdan Georgescu
// Created:     08/14/2003
// Version:     v0.1
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
//Modified by : Maithili  Paranjape
//Date	      : 09/09/04
//Functions modified : PruneModes
//Function added : SaveMymodes
//Version     : v0.2
/////////////////////////////////////////////////////////////////////////////

#include "mfams.h"
#include <lshreader.h>

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <functional>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

using namespace std;

namespace seg_meanshift {

FAMS::FAMS(const MeanShiftConfig &cfg, ProgressObserver *po)
	: config(cfg), po(po), progress(0.f), progress_old(0.f), lsh_(NULL)
{}

FAMS::~FAMS() {
}

#ifndef UNIX
#define drand48()    (rand() * 1.0 / RAND_MAX)
#endif

// Choose a subset of points on which to perform the mean shift operation
void FAMS::selectStartPoints(double percent, int jump) {
	if (datapoints.empty())
		return;

	size_t selectionSize;
	if (percent > 0.) {
		selectionSize = (size_t)(n_ * percent / 100.0);
	} else  {
		selectionSize = (size_t)ceil(n_ / (jump + 0.0));
	}

	if (selectionSize != startPoints.size()) {
		startPoints.resize(selectionSize);
		modes.resize(selectionSize);
	}

	if (percent > 0.) {
		for (size_t i = 0; i < startPoints.size();  i++)
			startPoints[i] = &datapoints[(int)(drand48() * n_) % n_];
	} else {
		for (size_t i = 0; i < startPoints.size(); i++)
			startPoints[i] = &datapoints[i * jump];
	}
}

void FAMS::importStartPoints(std::vector<Point> &points)
{
	/* add all points as starting points */
	startPoints.resize(points.size());
	for (size_t i = 0; i < points.size(); ++i)
		startPoints[i] = &points[i];
	modes.resize(startPoints.size());
}

void FAMS::ComputePilotPoint::operator()(const tbb::blocked_range<int> &r)
{
	const int thresh = (int)(fams.config.k * std::sqrt((float)fams.n_));
	const int win_j = 10, max_win = 7000;
	const int mwpwj = max_win / win_j;
	unsigned int nn;
	unsigned int wjd = (unsigned int)(win_j * fams.d_);

	LSHReader *lsh = NULL;
	if (fams.lsh_)
		lsh = new LSHReader(*fams.lsh_);

	int done = 0;
	for (int j = r.begin(); j != r.end(); ++j) {
		int numn = 0;
		int numns[mwpwj];
		memset(numns, 0, sizeof(numns));

		if (!lsh) {
			for (unsigned int i = 0; i < fams.n_; i++) {
				nn = fams.DistL1(fams.datapoints[j], fams.datapoints[i]) / wjd;
				if (nn < mwpwj)
					numns[nn]++;
			}
		} else {
			lsh->query(j);
			const std::vector<unsigned int> &lshResult = lsh->getResult();
			for (size_t i = 0; i < lshResult.size(); i++) {
				nn = fams.DistL1(fams.datapoints[j], fams.datapoints[lshResult[i]])
						/ wjd;
				if (nn < mwpwj)
					numns[nn]++;
			}
		}

		// determine distance to k-nearest neighbour
		for (nn = 0; nn < mwpwj; nn++) {
			numn += numns[nn];
			if (numn > thresh) {
				break;
			}
		}

		if (numn <= thresh) {
			dbg_noknn++;
		}

		fams.datapoints[j].window = (nn + 1) * wjd;
		fams.datapoints[j].weightdp2 = pow(
					FAMS_FLOAT_SHIFT / fams.datapoints[j].window,
					(fams.d_ + 2) * FAMS_ALPHA);
		if (weights) {
			fams.datapoints[j].weightdp2 *= (*weights)[j];
		}

		dbg_acc += fams.datapoints[j].window;

		if ((++done % (fams.n_ / 20)) == 0) {
			bool cont = fams.progressUpdate((float)done/(float)fams.n_ * 20.f,
				false);
			if (!cont) {
				bgLog("ComputePilot aborted.\n");
				return;
			}
			done = 0;
		}
	}
	fams.progressUpdate((float)done/(float)fams.n_ * 20.f, false);
	delete lsh;
}

// compute the pilot h_i's for the data points
bool FAMS::ComputePilot(vector<double> *weights) {
	bgLog("compute bandwidths...\n");

	if (config.use_LSH)
		assert(lsh_);

	ComputePilotPoint comp(*this, weights);
	tbb::parallel_reduce(tbb::blocked_range<int>(0, n_),
						 comp);

	cout << "Avg. window size: " << comp.dbg_acc / n_ << endl;
	bgLog("No kNN found for %2.2f%% of all points\n",
		  (float) comp.dbg_noknn / n_ * 100);

	return !(progress < 0.f); // in case of abort, progress is set to -1
}

// compute real bandwiths for selected points
void FAMS::ComputeRealBandwidths(unsigned int h) {
	const int thresh = (int)(config.k * std::sqrt((float)n_));
	const int    win_j = 10, max_win = 7000;
	unsigned int nn;
	unsigned int wjd;
	wjd =        (unsigned int)(win_j * d_);
	if (h == 0) {
		for (size_t j = 0; j < startPoints.size(); j++) {
			int numn = 0;
			int numns[max_win / win_j];
			memset(numns, 0, sizeof(numns));
			for (unsigned int i = 0; i < n_; i++) {
				nn = DistL1(*startPoints[j], datapoints[i]) / wjd;
				if (nn < max_win / win_j)
					numns[nn]++;
			}
			for (nn = 0; nn < max_win / win_j; nn++) {
				numn += numns[nn];
				if (numn > thresh) {
					break;
				}
			}
			startPoints[j]->window = (nn + 1) * win_j;
		}
	} else{
		for (size_t j = 0; j < startPoints.size(); j++) {
			startPoints[j]->window = h;
		}
	}
}

// compute the pilot h_i's for the data points
void FAMS::ComputeScores(float* scores, LSHReader &lsh, int L) {
	const int thresh = (int)(config.k * std::sqrt((float)n_));
	const int    win_j = 10, max_win = 7000;
	unsigned int nn;
	unsigned int wjd = (unsigned int)(win_j * d_);
	memset(scores, 0, L * sizeof(float));
	for (size_t j = 0; j < startPoints.size(); j++) {
		int nl = 0;
		int numns[max_win / win_j];
		memset(numns, 0, sizeof(numns));

		lsh.query(*startPoints[j]->data);
		const std::vector<unsigned int>& lshResult = lsh.getResult();
		const std::vector<int>& num_l = lsh.getNumByPartition();

		for (int i = 0; i < (int) lshResult.size(); i++) {
			nn = DistL1(*startPoints[j], datapoints[lshResult[i]]) / wjd;
			if (nn < max_win / win_j)
				numns[nn]++;

			if (i == (num_l[nl] - 1)) {
				// partition boundary
				/* current [0;i] represents the result after evaluating
				   nl partitions */

				// calculate distance to k-nearest neighbour in this result
				int numn = 0;
				for (nn = 0; nn < max_win / win_j; nn++) {
					numn += numns[nn];
					if (numn > thresh)
						break;
				}

				// assign score for this value of L and
				// any next partitions if they didn't add anything to the result
				for (; nl < L && (num_l[nl] - 1) == i; nl++) {
					assert(nl < L);
					scores[nl] += (float)(((nn + 1.0) * win_j) /
										   startPoints[j]->window);
				}
			}
		}
	}
	for (int j = 0; j < L; j++)
		scores[j] /= startPoints.size();
}


// perform a FAMS iteration
unsigned int FAMS::DoMSAdaptiveIteration(const std::vector<unsigned int> *res,
										 const std::vector<unsigned short> &old,
										 std::vector<unsigned short> &ret) const
{
	double total_weight = 0;
	double dist;
	std::vector<double> rr(d_, 0.);
	size_t nel = (res ? res->size() : n_);
	unsigned int crtH = 0;
	double       hmdist = 1e100;
	for (size_t i = 0; i < nel; i++) {
		const Point &ptp = (res ? datapoints[(*res)[i]] : datapoints[i]);
		if (DistL1Data(old, ptp, ptp.window, dist)) {
			double x = 1.0 - (dist / ptp.window);
			double w = ptp.weightdp2 * x * x;
			total_weight += w;
			for (size_t j = 0; j < ptp.data->size(); j++)
				rr[j] += (*ptp.data)[j] * w;
			if (dist < hmdist) {
				hmdist = dist;
				crtH   = ptp.window;
			}
		}
	}
	if (total_weight == 0) {
		return 0;
	}
	for (unsigned int i = 0; i < d_; i++)
		ret[i] = (unsigned short)(rr[i] / total_weight);

	return crtH;
}

void FAMS::MeanShiftPoint::operator()(const tbb::blocked_range<int> &r)
const
{
	LSHReader *lsh = NULL;
	if (fams.lsh_)
		lsh = new LSHReader(*fams.lsh_);

	// initialize mean vectors to zero
	std::vector<unsigned short>
			oldMean(fams.d_, 0),
			crtMean(fams.d_, 0);
	unsigned int *crtWindow;

	int done = 0;
	for (int jj = r.begin(); jj != r.end(); ++jj) {

		// update mode's window directly
		crtWindow  = &fams.modes[jj].window;
		// set initial values
		Point *p = fams.startPoints[jj];
		crtMean    = *p->data;
		*crtWindow = p->window;

		for (int iter = 0; oldMean != crtMean && (iter < FAMS_MAXITER);
			 iter++) {
			const std::vector<unsigned int> *lshResult = NULL;
			if (lsh) {
				Mode* solp = (Mode*)lsh->query(crtMean, &fams.modes[jj]);
				// test for solution cache hit, then if solution was yet found
				if (solp && !(solp->data.empty())) {
					/* early trajectory termination */
					fams.modes[jj] = *solp;
					break;
				}
				lsh->query(crtMean);
				lshResult = &lsh->getResult();
			}
			oldMean = crtMean;
			unsigned int newWindow =
				  fams.DoMSAdaptiveIteration(lshResult, oldMean, crtMean);
			if (!newWindow) {
				// oldMean is final mean -> break loop
				break;
			}
			*crtWindow = newWindow;
		}

		// algorithm converged, store result if we do not already know it
		if (fams.modes[jj].data.empty()) {
			fams.modes[jj].data = crtMean;
		}

		// progress reporting
		if (fams.startPoints.size() < 80 ||
			(++done % (fams.startPoints.size() / 80)) == 0) {
			bool cont = fams.progressUpdate((float)done/
											(float)fams.startPoints.size()*80.f,
											false);
			if (!cont) {
				delete lsh;
				bgLog("FinishFAMS aborted.\n");
				return;
			}
			done = 0;
		}
	}
	fams.progressUpdate((float)done/(float)fams.startPoints.size()*80.f, false);

	delete lsh;
}

// perform FAMS starting from a subset of the data points.
// return true on successful finish (not cancelled by ProgressObserver)
bool FAMS::finishFAMS() {
	bgLog(" Start MS iterations\n");

	if (config.use_LSH)
		assert(lsh_);

	// hack: no parallel LSH
	if (config.use_LSH) {
		bgLog("*** HACK: no tbb for LSH-enabled mean shift\n");
		MeanShiftPoint worker(*this);
		worker(tbb::blocked_range<int>(0, startPoints.size()));
	} else {
		tbb::parallel_for(tbb::blocked_range<int>(0, startPoints.size()),
						  MeanShiftPoint(*this));
	}

	delete lsh_; // cleanup
	lsh_ = NULL;
	bgLog("done.\n");
	return !(progress < 0.f); // in case of abort, progress is set to -1
}

// main function to find K and L
std::pair<int,int> FAMS::FindKL() {
	int Kmin = config.Kmin, Kmax = config.K, Kjump = config.Kjump;
	int Lmax = config.L, k = config.K;
	float width = config.bandwidth, epsilon = config.epsilon;

	bgLog("Find optimal K and L, K=%d:%d:%d, Lmax=%d, k=%d, Err=%.2g\n",
		  Kmin, Kjump, Kmax, Lmax, k, epsilon);

	if (datapoints.empty()) {
		bgLog("Load points first\n");
		return make_pair(0, 0);
	}

	int hWidth   = 0;
	if (width > 0.f) {
		hWidth   = value2ushort<int>(width);
	}
	epsilon += 1;

	/// sanity checks
	assert(Kmin <= Kmax);
	assert(Lmax >= 1);

	// select points on which test is run
	selectStartPoints(FAMS_FKL_NEL * 100.0 / n_, 0);

	// compute bandwidths for selected points
	ComputeRealBandwidths(hWidth);

	// start finding the correct l for each k
	// scores for 10 trials runs per L
	std::vector<float> scores(FAMS_FKL_TIMES * Lmax);
	int   Lcrt, Kcrt;

	int nBest;
	std::vector<int> LBest(Kmax); /// contains the best L for each tested K
	std::vector<int> KBest(Kmax); /// contains the actual value of K for each tested K

	int ntimes, is;
	Lcrt = Lmax;
	bgLog(" find valid pairs.. ");
	/// for each K...
	for (Kcrt = Kmax, nBest = 0; Kcrt >= Kmin; Kcrt -= Kjump, nBest++) {
		// do iterations for current K and L = 1...Lcrt
		for (ntimes = 0; ntimes < FAMS_FKL_TIMES; ntimes++)
			DoFindKLIteration(Kcrt, Lcrt, &scores[ntimes * Lcrt]);

		// get best L for current k
		KBest[nBest] = Kcrt;
		LBest[nBest] = -1;
		for (is = 0; is < Lcrt; is++) {
			// find worst error with this L
			for (ntimes = 1; ntimes < FAMS_FKL_TIMES; ntimes++) {
				if (scores[is] < scores[ntimes * Lcrt + is])
					scores[is] = scores[ntimes * Lcrt + is];
			}
			if (scores[is] < epsilon) {
				LBest[nBest] = is + 1;
				break; /// stop at first match
			}
		}
		bool cont = progressUpdate(50.f * (Kmax-Kcrt)/(Kmax-Kmin));
		if (!cont) {
			bgLog("FindKL aborted\n");
			return std::make_pair(0, 0);
		}

		// update Lcrt to reduce running time!
		// (-> next lower K wont give any better results with a much higher L)
		if (LBest[nBest] > 0)
			Lcrt = min(LBest[nBest] + 2, Lmax);
	}
	bgLog("done\n");

	//start finding the pair with best running time
	int64 run_times[FAMS_FKL_TIMES];
	int iBest = -1;
	int i;
	int64 timeBest = -1;
	bgLog(" select best pair\n");
	for (i = 0; i < nBest; i++) {
		bool cont = progressUpdate(50.f + 50.f * i/nBest);
		if (!cont) {
			bgLog("FindKL aborted\n");
			return std::make_pair(0, 0);
		}

		if (LBest[i] <= 0)
			continue;
		for (ntimes = 0; ntimes < FAMS_FKL_TIMES; ntimes++)
			run_times[ntimes] =
				DoFindKLIteration(KBest[i], LBest[i], &scores[ntimes * Lcrt]);
		sort(&run_times[0], &run_times[FAMS_FKL_TIMES]);
		// compare with median
		if ((timeBest == -1) || (timeBest > run_times[FAMS_FKL_TIMES / 2])) {
			iBest    = i;
			timeBest = run_times[FAMS_FKL_TIMES / 2];
		}
		bgLog("  K=%d L=%d time: %g\n", KBest[i], LBest[i],
			  run_times[FAMS_FKL_TIMES / 2]);
	}
	bgLog("done\n");

	if (iBest != -1) {
		return std::make_pair(KBest[iBest], LBest[iBest]);
	} else {
		bgLog("No valid pairs found.\n");
		return std::make_pair(0, 0);
	}
}


int64 FAMS::DoFindKLIteration(int K, int L, float* scores) {
	LSH lsh(dataholder, d_, K, L);
	LSHReader lshreader(lsh);

	// Compute Scores
	int64 ticks = cv::getTickCount();
	ComputeScores(scores, lshreader, L);
	return cv::getTickCount() - ticks;
}

// initialize lsh, bandwidths
bool FAMS::prepareFAMS(vector<double> *bandwidths) {
	assert(!datapoints.empty());

	if (config.use_LSH) {
		bgLog("Running FAMS with K=%d L=%d\n", config.K, config.L);
		lsh_ = new LSH(dataholder, d_, config.K, config.L);
	} else {
		bgLog("Running FAMS without LSH (try --useLSH)\n");
	}

	//Compute pilot if necessary
	bgLog(" Run pilot ");
	bool cont = true;
#ifdef WITH_SEG_FELZENSZWALB
	bool adaptive = (config.bandwidth <= 0. ||
					 bandwidths == NULL || config.sp_weight == 2);
#else
	bool adaptive = (config.bandwidth <= 0. || bandwidths == NULL);
#endif
	if (adaptive) {  // adaptive bandwidths
		bgLog("adaptive...");
		cont = ComputePilot(bandwidths);
	} else if (bandwidths != NULL) {  // preset bandwidths
		bgLog("fixed bandwidth (local value)...");
		assert(bandwidths->size() == n_);
		cout << "maxVal_ = " << maxVal_ << endl;
		cout << "minVal_ = " << minVal_ << endl;
		for (unsigned int i = 0; i < n_; i++) {
			double width = bandwidths->at(i);
			unsigned int hWidth = value2ushort<unsigned int>(width);

			datapoints[i].window = hWidth;
			datapoints[i].weightdp2 = pow(
						FAMS_FLOAT_SHIFT / datapoints[i].window,
						(d_ + 2) * FAMS_ALPHA);
		}
	} else {  // fixed bandwidth for all points
		bgLog("fixed bandwidth (global value)...");
		int hWidth = value2ushort<int>(config.bandwidth);
		unsigned int hwd = (unsigned int)(hWidth * d_);
		cout << "Window size: " << hwd << endl;
		for (unsigned int i = 0; i < n_; i++) {
			datapoints[i].window    = hwd;
			datapoints[i].weightdp2 = 1;
		}
	}

	if (!cont) {
		delete lsh_;
		lsh_ = NULL;
	}
	bgLog("done.\n");
	return cont;
}

bool FAMS::progressUpdate(float percent, bool absolute)
{
	if (!po && config.verbosity < 1)
		return true;

	tbb::mutex::scoped_lock(progressMutex);
	if (absolute)
		progress = percent;
	else
		progress += percent;
	if (progress > progress_old + 0.1f) {
		if (config.verbosity > 1) {
			std::cerr << "\r" << progress << " %          \r";
			std::cerr.flush();
		}
		progress_old = progress;
	}

	if (!po)
		return true;
	bool cont = po->update(progress / 100);
	if (!cont)
		progress = -1.f;
	return cont;
}

}
