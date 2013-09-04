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

#include "mfams.h"
#include <lshreader.h>


using namespace std;

FAMS::FAMS(const vole::MeanShiftConfig &cfg, vole::ProgressObserver *po)
	: hasPoints_(0), nsel_(0), npm_(0), config(cfg),
	  progressObserver(po), progress(0.f), progress_old(0.f), lsh_(NULL)
{}

FAMS::~FAMS() {
	CleanPoints();
	CleanPrunedModes();
}

void FAMS::CleanPoints() {
	if (!hasPoints_)
		return;

	delete [] points_;
	delete [] data_;
	hasPoints_ = 0;
}

void FAMS::CleanPrunedModes() {
	if (!npm_)
		return;

	delete [] prunedmodes_;
	delete [] nprunedmodes_;
	npm_ = 0;
}

// Choose a subset of points on which to perform the mean shift operation
void FAMS::SelectMsPoints(double percent, int jump) {
	if (!hasPoints_)
		return;

	int tsel;
	if (percent > 0.) {
		tsel = (int)(n_ * percent / 100.0);
	} else  {
		tsel = (int)ceil(n_ / (jump + 0.0));
	}

	if (tsel != nsel_) {
		nsel_   = tsel;
		psel_.resize(nsel_);
		modes_.resize(nsel_ * d_);
		hmodes_.resize(nsel_);
	}

	if (percent > 0.) {
		for (int i = 0; i < nsel_;  i++)
			psel_[i] = &points_[std::min(n_ - 1, (int)(drand48() * n_))];
	} else {
		for (int i = 0; i < nsel_; i++)
			psel_[i] = &points_[i * jump];
	}
}

void FAMS::ImportMsPoints(std::vector<fams_point> &points) {
	nsel_ = points.size();
	psel_.resize(nsel_);
	modes_.resize(nsel_ * d_);
	hmodes_.resize(nsel_);
	for (int i = 0; i < nsel_; i++)
		psel_[i] = &points[i];
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
	for (size_t j = r.begin(); j != r.end(); ++j) {
		int numn = 0;
		int numns[mwpwj];
		memset(numns, 0, sizeof(numns));

		if (!lsh) {
			for (int i = 0; i < fams.n_; i++) {
				nn = fams.DistL1(fams.points_[j], fams.points_[i]) / wjd;
				if (nn < mwpwj)
					numns[nn]++;
			}
		} else {
			lsh->query(j);
			const std::vector<unsigned int> &lshResult = lsh->getResult();
			for (int i = 0; i < lshResult.size(); i++) {
				nn = fams.DistL1(fams.points_[j], fams.points_[lshResult[i]])
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

		fams.points_[j].window_ = (nn + 1) * wjd;
		fams.points_[j].weightdp2_ = pow(
					FAMS_FLOAT_SHIFT / fams.points_[j].window_,
					(fams.d_ + 2) * FAMS_ALPHA);
		if (weights) {
			fams.points_[j].weightdp2_ *= (*weights)[j];
		}

		dbg_acc += fams.points_[j].window_;

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
	int          i, j;
	unsigned int nn;
	unsigned int wjd;
	wjd =        (unsigned int)(win_j * d_);
	if (h == 0) {
		for (j = 0; j < nsel_; j++) {
			int numn = 0;
			int numns[max_win / win_j];
			memset(numns, 0, sizeof(numns));
			for (i = 0; i < n_; i++) {
				nn = DistL1(*psel_[j], points_[i]) / wjd;
				if (nn < max_win / win_j)
					numns[nn]++;
			}
			for (nn = 0; nn < max_win / win_j; nn++) {
				numn += numns[nn];
				if (numn > thresh) {
					break;
				}
			}
			psel_[j]->window_ = (nn + 1) * win_j;
		}
	} else{
		for (j = 0; j < nsel_; j++) {
			psel_[j]->window_ = h;
		}
	}
}

// compute the pilot h_i's for the data points
void FAMS::ComputeScores(float* scores, LSHReader &lsh, int L) {
	const int thresh = (int)(config.k * std::sqrt((float)n_));
	const int    win_j = 10, max_win = 7000;
	int          j;
	unsigned int nn;
	unsigned int wjd = (unsigned int)(win_j * d_);
	memset(scores, 0, L * sizeof(float));
	for (j = 0; j < nsel_; j++) {
		int nl = 0;
		int numns[max_win / win_j];
		memset(numns, 0, sizeof(numns));

		lsh.query(psel_[j]->data_);
		const std::vector<unsigned int>& lshResult = lsh.getResult();
		const std::vector<int>& num_l = lsh.getNumByPartition();

		for (int i = 0; i < (int) lshResult.size(); i++) {
			nn = DistL1(*psel_[j], points_[lshResult[i]]) / wjd;
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
										   psel_[j]->window_);
				}
			}
		}
	}
	for (j = 0; j < L; j++)
		scores[j] /= nsel_;
}


// perform a FAMS iteration
unsigned int FAMS::DoMSAdaptiveIteration(
			const std::vector<unsigned int> *res,
			unsigned short *old, unsigned short *ret) const
{
	double total_weight = 0;
	int    i, j;
	double dist;
	std::vector<double> rr(d_, 0.);
	int nel = (config.use_LSH ? res->size() : n_);
	unsigned int crtH = 0;
	double       hmdist = 1e100;
	for (i = 0; i < nel; i++) {
		fams_point &ptp = (config.use_LSH ? points_[(*res)[i]] : points_[i]);
		if (DistL1Data(old, ptp, ptp.window_, dist)) {
			double x = 1.0 - (dist / ptp.window_);
			double w = ptp.weightdp2_ * x * x;
			total_weight += w;
			for (j = 0; j < d_; j++)
				rr[j] += ptp.data_[j] * w;
			if (dist < hmdist) {
				hmdist = dist;
				crtH   = ptp.window_;
			}
		}
	}
	if (total_weight == 0) {
		return 0;
	}
	for (i = 0; i < d_; i++)
		ret[i] = (unsigned short)(rr[i] / total_weight);

	return crtH;
}

void FAMS::MeanShiftPoint::operator()(const tbb::blocked_range<int> &r)
const
{
	LSHReader *lsh = NULL;
	if (fams.lsh_)
		lsh = new LSHReader(*fams.lsh_);

	unsigned short *oldMean;
	unsigned short *crtMean;
	oldMean = new unsigned short[fams.d_];
	crtMean = new unsigned short[fams.d_];
	memset(oldMean, 0, sizeof(*oldMean) * fams.d_);
	memset(crtMean, 0, sizeof(*crtMean) * fams.d_);
	unsigned int   newH;
	unsigned int   *crtH;

	struct mode {
		unsigned short *m;
		unsigned int *h;
	};
	struct mode *tMode = new struct mode[fams.n_];

	int done = 0;
	for (int jj = r.begin(); jj != r.end(); ++jj) {

		fams_point *currentpt = fams.psel_[jj];
		memcpy(crtMean, currentpt->data_, fams.dataSize_);
		crtH      = &fams.hmodes_[jj];
		*crtH     = currentpt->window_;
		tMode[jj].m = (unsigned short*)1;

		for (int iter = 0; fams.NotEq(oldMean, crtMean) &&
			 (iter < FAMS_MAXITER); iter++) {
			const std::vector<unsigned int> *lshResult = NULL;
			if (lsh) {
				struct mode* solp;
				solp = (struct mode*) lsh->query(crtMean, &tMode[jj]);
				// NULL means no solution cache hit
				// 1 means there's no actual solution yet
				if (solp != NULL && solp->m != (unsigned short*) 1) {
					// early trajectory termination, break loop

					tMode[jj].m = &fams.modes_[jj * fams.d_];
					memcpy(tMode[jj].m, solp->m, fams.dataSize_);

					tMode[jj].h = &fams.hmodes_[jj];
					*(tMode[jj].h) = *(solp->h);
					break;
				}
				lsh->query(crtMean);
				lshResult = &lsh->getResult();
			}
			memcpy(oldMean, crtMean, fams.dataSize_);
			if (!(newH =
				  fams.DoMSAdaptiveIteration(lshResult, oldMean, crtMean))) {
				// oldMean is final mean -> break loop
				memcpy(crtMean, oldMean, fams.dataSize_);
				break;
			}
			*crtH = newH;
		}

		if (tMode[jj].m == (unsigned short*)1) {
			tMode[jj].m = &fams.modes_[jj * fams.d_];
			memcpy(tMode[jj].m, crtMean, fams.dataSize_);
			tMode[jj].h = &fams.hmodes_[jj];
		}

		if (fams.nsel_ < 80 || (++done % (fams.nsel_ / 80)) == 0) {
			bool cont = fams.progressUpdate((float)done/
											(float)fams.nsel_ * 80.f, false);
			if (!cont) {
				delete [] oldMean;
				delete [] crtMean;
				delete [] tMode;
				delete lsh;
				bgLog("FinishFAMS aborted.\n");
				return;
			}
			done = 0;
		}
	}
	fams.progressUpdate((float)done/(float)fams.nsel_ * 80.f, false);

	delete[] oldMean;
	delete[] crtMean;
	delete[] tMode;
	delete lsh;
}

// perform FAMS starting from a subset of the data points.
// return true on successful finish (not cancelled by ProgressObserver)
bool FAMS::FinishFAMS() {
	bgLog(" Start MS iterations\n");

	if (config.use_LSH)
		assert(lsh_);

	// hack: no parallel LSH
	if (config.use_LSH) {
		bgLog("*** HACK: no tbb for LSH-enabled mean shift\n");
		MeanShiftPoint worker(*this);
		worker(tbb::blocked_range<int>(0, nsel_));
	} else {
		tbb::parallel_for(tbb::blocked_range<int>(0, nsel_),
						  MeanShiftPoint(*this));
	}

	delete lsh_; // cleanup
	lsh_ = NULL;
	bgLog("done.\n");
	return !(progress < 0.f); // in case of abort, progress is set to -1
}


void FAMS::PruneModes() {
	int hprune = FAMS_PRUNE_WINDOW;
	int npmin = config.pruneMinN;
	// compute jump
	int jm = (int)ceil(((double)nsel_) / FAMS_PRUNE_MAXP);

	bgLog(" Join Modes with adaptive h/%d, min pt=%d, jump=%d"
		  "(-> looking at %d modes)\n", (int) pow(2.f,FAMS_PRUNE_HDIV), npmin,
		  jm, nsel_/jm);
	bgLog("            pass 1");
	if (nsel_ < 1)
		return;
	hprune *= d_;

	int            *mcount, *mcount2, *mycount, *mcountsp;
	float          *cmodes, *ctmodes, *cmodes2;
	unsigned short *pmodes;
	double         cminDist, cdist;
	int            iminDist, cref;
	int            oldmaxm;
	unsigned char  *invalidm;
	invalidm = new unsigned char[nsel_];
	mcount   = new int[nsel_];
	mcountsp   = new int[nsel_];
	mycount  = new int[nsel_];
	cmodes   = new float[d_ * nsel_];
	// to save the image data after pruning of modes.
	mymodes     = new float[d_ * nsel_];
	testmymodes = new float[d_ * nsel_];
	tmpmymodes  = new float[d_ * nsel_];
	indmymodes  = new int[nsel_];

	int i, j, k, cd, cm, maxm, idx, idx1, idx2, idx3, cd1, cd2;
	idx  = 0;
	idx1 = 0;
	idx2 = 0;
	idx3 = 0;


	memset(mcount, 0, nsel_ * sizeof(int));
	memset(mcountsp, 0, nsel_ * sizeof(int));
	memset(invalidm, 0, nsel_ * sizeof(unsigned char));

	memset(mycount, 0, nsel_ * sizeof(int));


	// copy the image data before mode pruning.
	for (j = 0; j < nsel_; j++) {
		for (i = 0; i < d_; i++) {
			mymodes[idx] = modes_[idx];
			idx++;
		}
	}


	// set first mode
	for (cd = 0; cd < d_; cd++) {
		cmodes[cd] = modes_[cd];
	}
	mcount[0] = 1;
	mcountsp[0] = (spsizes.empty() ? 1 : spsizes[0]);
	maxm      = 1;

	int myPt = FAMS_PRUNE_MAXP / 10;

	int invalidc = 0;


	for (cm = 1; cm < nsel_; cm += jm) {
		if ((cm % myPt) == 0)
			bgLog(".");

		pmodes = &modes_[cm * d_];

		// compute closest mode
		cminDist = d_ * 1e7;
		iminDist = -1;
		for (cref = 0; cref < maxm; cref++) {
			if (invalidm[cref])
				continue;
			cdist   = 0;
			ctmodes = cmodes + cref * d_;
			for (cd = 0; cd < d_; cd++)
				cdist += fabs(ctmodes[cd] / mcount[cref] - pmodes[cd]);
			if (cdist < cminDist) {
				cminDist = cdist;
				iminDist = cref;
			}
		}
		// join

		// good & cheap indicator for serious failure in DoFAMS()
		assert(hmodes_[cm] > 0);

		hprune = hmodes_[cm] >> FAMS_PRUNE_HDIV;

		if (cminDist < hprune) {
			// already in, just add
			// add mode of current point (cm) to closed known mode
			// put result in mymodes

			for (cd = 0; cd < d_; cd++) {
				cmodes[iminDist * d_ + cd] += pmodes[cd];
				mymodes[cm * d_ + cd]       = cmodes[iminDist * d_ + cd];
			}
			// increase counter
			mcount[iminDist] += 1;
			mcountsp[iminDist] += (spsizes.empty() ? 1 : spsizes[cm]);

			// "normalize" entry in mymodes
			for (cd1 = 0; cd1 < d_; cd1++) {
				mymodes[cm * d_ +
						cd1] = mymodes[cm * d_ + cd1] / mcount[iminDist];
			}
		} else{
			// new mode, create
			// closest known mode is to far away, assume new mode
			for (cd = 0; cd < d_; cd++) {
				cmodes[maxm * d_ + cd] = pmodes[cd];
			}


			mcount[maxm] = 1;
			mcountsp[maxm] = (spsizes.empty() ? 1 : spsizes[cm]);

			maxm += 1;
		}
		// check for valid modes
		// invalidate (delete) modes with few members
		if (maxm > 2000) {
			invalidc = 0;
			for (i = 0; i < maxm; i++) {
				if (mcount[i] < 3) {
					invalidm[i] = 1;
					invalidc++;
				}
			}
		}
	}

	oldmaxm = maxm;
	bgLog("done (%d modes left, %d of them have been invalidated)\n", maxm,
		  invalidc);
	bgLog("            pass 2");

	// put the modes in the order of importance (count)
	vector<pair<int, int> > xtemp(maxm);
	for (i = 0; i < maxm; ++i) {
		xtemp[i] = make_pair(mcountsp[i], i);
	}
	sort(xtemp.begin(), xtemp.end());

	// find number of relevant modes
	int nrel = 1;
	for (i = maxm - 2; i >= 0; --i) {
		if (xtemp[i].first >= npmin)
			nrel++;
		else
			break;
	}

	bgLog("ignoring %d modes smaller than %d points\n", (maxm - nrel), npmin);
	if (nrel > FAMS_PRUNE_MAXM) {
		bgLog("exceeded FAMS_PRUNE_MAXM, only keeping %d modes\n",
			  FAMS_PRUNE_MAXM);
	}

	// HACK
	if (!spsizes.empty())
		npmin = 1;

	nrel = std::min(nrel, FAMS_PRUNE_MAXM);

	// rearange only relevant modes
	mcount2 = new int[nrel];
	cmodes2 = new float[d_ * nrel];

	for (i = 0; i < nrel; i++) {
		cm         = xtemp[maxm - i - 1].second; // index
		mcount2[i] = mcount[cm];
		memcpy(cmodes2 + i * d_, cmodes + cm * d_, d_ * sizeof(float));
	}

	delete [] cmodes;
	memset(mcount, 0, nsel_ * sizeof(int));

	maxm = nrel;

	//Computation of the closet mode for pixel zero
	bool flag;
	int  num;
	flag     = false;
	num      = 1;
	pmodes   = &modes_[0];
	cminDist = d_ * 1e7;
	iminDist = -1;

	for (cref = 0; cref < maxm; cref++) {
		cdist   = 0;
		ctmodes = cmodes2 + cref * d_;
		for (cd = 0; cd < d_; cd++)
			cdist += fabs(ctmodes[cd] / mcount2[cref] - pmodes[cd]);
		if (cdist < cminDist) {
			cminDist = cdist;
			iminDist = cref;
			flag     = true;
		}
	}
	// if the closest mode is found the mode number is saved.
	indmymodes[0] = (flag == true ? iminDist : 0);

	myPt = max(1, nsel_ / 10); /// minimum 1 to prevent division by zero
	for (cm = 1; cm < nsel_; cm++) {
		if ((cm % myPt) == 0)
			bgLog(".");

		/*if (mcount[cm])
		   continue;*/

		pmodes = &modes_[cm * d_];

		// compute closest mode
		cminDist = d_ * 1e7;
		iminDist = -1;
		for (cref = 0; cref < maxm; cref++) {
			cdist   = 0;
			ctmodes = cmodes2 + cref * d_;
			for (cd = 0; cd < d_; cd++)
				cdist += fabs(ctmodes[cd] / mcount2[cref] - pmodes[cd]);
			if (cdist < cminDist) {
				cminDist = cdist;
				iminDist = cref;
			}
		}
		// join
		hprune = hmodes_[cm] >> FAMS_PRUNE_HDIV;
		/* if the closet mode is found, the avg pixel value for that mode is
		 * calculated by divding mymodes with mcount(the number of pixels
		 * having that mode)*/
		if (iminDist >= 0) {
			// aready in, just add
			for (cd = 0; cd < d_; cd++) {
				cmodes2[iminDist * d_ + cd] += pmodes[cd];
				mymodes[cm * d_ + cd]        = cmodes2[iminDist * d_ + cd];
			}
			indmymodes[cm] = iminDist;

			mcount2[iminDist] += 1;
			for (cd2 = 0; cd2 < d_; cd2++) {
				mymodes[cm * d_ +
						cd2] = mymodes[cm * d_ + cd2] / mcount2[iminDist];
			}
		} else {
			indmymodes[cm] = 8888; /// ?!?!
		}
	}

	/*Copy the current values all the pixels as per their mode to array
	 *'testmymodes'*/
	for (j = 0; j < nsel_; j++) {
		for (i = 0; i < d_; i++) {
			testmymodes[idx3] = mymodes[idx3];
			tmpmymodes[idx3]  = 0.0;
			idx3++;
		}
	}
	/*For all the pixels having same mode, an average value of that mode is
	 * assigned */
	for (i = 0; i < maxm; i++) {
		for (j = 0; j < nsel_; j++) {
			if (i == indmymodes[j]) {
				for (k = 0; k < d_; k++)
					tmpmymodes[i * d_ + k] += testmymodes[j * d_ + k];

				mycount[i] += 1;
			}
		}
	}

	for (i = 0; i < maxm; i++) {
		for (k = 0; k < d_; k++)
			tmpmymodes[i * d_ + k] = tmpmymodes[i * d_ + k] / mycount[i];
	}


	// All the pixels having same mode have the same pixel value.
	for (i = 0; i < maxm; i++) {
		for (j = 0; j < nsel_; j++) {
			if (i == indmymodes[j]) {
				for (k = 0; k < d_; k++)
					testmymodes[j * d_ + k] = tmpmymodes[i * d_ + k];
			} else if (indmymodes[j] == 8888) {
				for (k = 0; k < d_; k++)
					testmymodes[j * d_ + k] = 0.0;
			}
		}
	}

	// sort modes in the order of importance (count)
	xtemp.resize(maxm);
	for (i = 0; i < maxm; ++i) {
		xtemp[i] = make_pair(mcount2[i], i);
	}
	sort(xtemp.begin(), xtemp.end());

	// find number of relevant modes
	nrel = 1;
	for (i = maxm - 2; i >= 0; i--) {
		if (xtemp[i].first >= npmin)
			nrel++;
		else
			break;
	}

	bgLog("once more ignoring %d modes smaller than %d points\n",
		  (maxm - nrel), npmin);

	CleanPrunedModes();
	prunedmodes_  = new unsigned short[d_ * nrel];
	nprunedmodes_ = new int[nrel];
	unsigned short* cpm;
	npm_ = nrel;

	cpm = prunedmodes_;
	for (i = 0; i < npm_; i++) {
		nprunedmodes_[i] = xtemp[maxm - i - 1].first;
		cm = xtemp[maxm - i - 1].second;
		for (cd = 0; cd < d_; cd++) {
			*(cpm++) = (unsigned short)(cmodes2[cm * d_ + cd] / mcount2[cm]);
		}
	}

	delete [] cmodes2;
	delete [] mcount2;
	delete [] mcount;

	bgLog("done\n");
}


// main function to find K and L
std::pair<int,int> FAMS::FindKL() {
	int Kmin = config.Kmin, Kmax = config.K, Kjump = config.Kjump;
	int Lmax = config.L, k = config.K;
	float width = config.bandwidth, epsilon = config.epsilon;

	bgLog("Find optimal K and L, K=%d:%d:%d, Lmax=%d, k=%d, Err=%.2g\n",
		  Kmin, Kjump, Kmax, Lmax, k, epsilon);

	if (hasPoints_ == 0) {
		bgLog("Load points first\n");
		return make_pair(0, 0);
	}

	bool adaptive = true;
	int hWidth   = 0;
	if (width > 0.f) {
		adaptive = false;
		hWidth   = (int)(65535.0 * (width) / (maxVal_ - minVal_));
	}
	epsilon += 1;

	/// sanity checks
	assert(Kmin <= Kmax);
	assert(Lmax >= 1);

	// select points on which test is run
	SelectMsPoints(FAMS_FKL_NEL * 100.0 / n_, 0);

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
	LSH lsh(data_, n_, d_, K, L);
	LSHReader lshreader(lsh);

	// Compute Scores
	int64 ticks = cv::getTickCount();
	ComputeScores(scores, lshreader, L);
	return cv::getTickCount() - ticks;
}

// initialize lsh, bandwidths
bool FAMS::PrepareFAMS(vector<double> *bandwidths) {
	assert(hasPoints_);

	if (config.use_LSH) {
		bgLog("Running FAMS with K=%d L=%d\n", config.K, config.L);
		lsh_ = new LSH(data_, n_, d_, config.K, config.L);
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
		for (int i = 0; i < n_; i++) {
			double width = bandwidths->at(i);
			unsigned int hWidth =
					(unsigned int)(65535.0 * width / (maxVal_ - minVal_));

			points_[i].window_ = hWidth;
			points_[i].weightdp2_ = pow(
						FAMS_FLOAT_SHIFT / points_[i].window_,
						(d_ + 2) * FAMS_ALPHA);
		}
	} else {  // fixed bandwidth for all points
		bgLog("fixed bandwidth (global value)...");
		int hWidth = (int)(65535.0 * (config.bandwidth)
						   / (maxVal_ - minVal_));
		unsigned int hwd = (unsigned int)(hWidth * d_);
		cout << "Window size: " << hwd << endl;
		for (int i = 0; i < n_; i++) {
			points_[i].window_    = hwd;
			points_[i].weightdp2_ = 1;
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
	if (progressObserver == NULL && config.verbosity < 1)
		return true;

	tbb::mutex::scoped_lock(progressMutex);
	if (absolute)
		progress = percent;
	else
		progress += percent;
	if (progress > progress_old + 0.1f) {
		std::cerr << "\r" << progress << " %          \r";
		std::cerr.flush();
		progress_old = progress;
	}

	if (progressObserver == NULL)
		return true;
	bool cont = progressObserver->update((int)progress);
	if (!cont)
		progress = -1.f;
	return cont;
}
