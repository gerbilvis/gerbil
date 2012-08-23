/*	
	Copyright(c) 2011 Daniel Danner,
	Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "probshift.h"

#include <multi_img.h>
#include <highgui.h>
#include <iostream>
#include <cstdio>
#include <queue>
#include <numeric>
#include <iostream>
#include <fstream>
#include <lsh.h>

#include <slepceps.h>
#include <petscmat.h>

#include <boost/math/distributions/binomial.hpp>

#include "evrot.cpp"

#include <meanshift.h>
#include <meanshift_config.h>
#include <labeling.h>

#define PETSCCHKERR(ierr) CHKERRABORT(PETSC_COMM_WORLD, ierr)

// granularity of distance bucket-sort
#define PROBSHIFT_DISTBUCKETS 7000

// maximum window size for neighborhoods (percentage of max distance)
#define PROBSHIFT_MAXWIN 1.0f

// minimum influence neighbors to start with (default: 25)
#define PROBSHIFT_WIN_SIZE 25

// maximum window size for influence neighborhoods
#define PROBSHIFT_WIN_SIZE_MAX 500u

// grow window size in this step (default 2)
#define PROBSHIFT_WINJUMP 2

// threshold for convergence in random walks (default 0.0005)
#define PROBSHIFT_RW_DIFFTHRESH 0.005

// threshold increase (factor) after 500 iterations
#define PROBSHIFT_RW_THRESHINCR 10

// maximum number of iterations with useMeanShift and msClusts
#define PROBSHIFT_MSPOSTPROC_MAXITER 100

namespace vole {

using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using std::flush;
using boost::math::binomial_distribution;

typedef multi_img::Pixel Pixel;
typedef multi_img::Value Value;


cv::Mat1s ProbShift::execute(const multi_img& input, const std::string& base, ProgressObserver *progress) {
	std::cout << "Probabilistic Shift Segmentation" << std::endl;

	progressObserver = progress;

	unsigned int dims = input.size();
	pair<multi_img::Value, multi_img::Value> minmax = input.data_range();
	double maxL2 = sqrt(pow((double) (minmax.second - minmax.first), 2) * dims);
	unsigned int npoints = input.width * input.height;

	// make sure pixel cache is up to date
	input.rebuildPixels();

	if (!config.loadModes.empty()) {
		// external modes -> do MeanShift / assignModes right here
		vector<unsigned int> armode = loadModes(config.loadModes);

		if (armode.size() != input.width * input.height) {
			cerr << "Error: input dimensions do not match loaded mode assignments" << endl;
			return cv::Mat1s();
		}

		if (config.useMeanShift)
			armode = meanShiftPostProcess(armode, input);

		return assignLabels(armode, input.width, input.height);
	}

	// ProbShift

	std::cerr << "Determining influence neighborhoods and shift vectors" << std::endl;

	// maps each point to the points in its influence neighborhood
	vector< vector<unsigned int> > infNeigh(npoints);

	// maps each point to the influence neighborhoods it appears in (only for printing statistics)
	vector< vector<unsigned int> > infNeighRev(npoints);

	// wether a point had an isotropic neighborhood at least once (only for printing statistics)
	vector<bool> hasAnyIsotropy(npoints, false);

	vector< Mat_<double> > votes(npoints);
	for (unsigned int i = 0; i < npoints; ++i)
		votes[i] = Mat_<double>::zeros(dims, 1);
	vector<bool> hasVote(npoints, false);

	int tooSmallNeighborhood = 0;
	int maxwin = min(PROBSHIFT_WIN_SIZE_MAX, npoints);

	// TODO: make LSH feed on multi_img directly?
	unsigned short *data = NULL;
	cv::flann::Index_<multi_img::Value> *flannIndex = NULL;

	Mat_<multi_img::Value> pointsMat(npoints, dims);
	Mat_<float> queriesDists(npoints, maxwin);
	Mat_<int> queriesIndices(npoints, maxwin, -1);

	if (config.useLSH) {
		// prepare input for LSH
		data = input.export_interleaved(true);
	} else {
		// prepare kdTree
		for (int i = 0; i < (int) npoints; ++i) {
			for (int d = 0; d < (int) dims; ++d)
				pointsMat(i, d) = input.atIndex(i)[d];
		}

		cout << "Building FLANN index..." << flush;
		cvflann::KDTreeIndexParams paramIdx;	// TODO: only OpenCV 2.3.1?
//		cv::flann::LinearIndexParams paramIdx;  // did not work anymore with 2.3.1
		flannIndex = new cv::flann::Index_<multi_img::Value>(pointsMat, paramIdx);
		cout << "done" << endl;

		cout << "Querying all neighborhoods at once..." << flush;
		cvflann::SearchParams paramSearch;
		flannIndex->knnSearch(pointsMat, queriesIndices, queriesDists, maxwin, paramSearch);
		cout << "done" << endl;
	}
#pragma omp parallel
	{
	// thread-local LSH instance
	LSH *lsh = NULL;
	if (config.useLSH)
		lsh = new LSH(data, npoints, input.size(), config.lshK, config.lshL);

	// thread-local votes
	vector< Mat_<double> > localvotes(npoints);
	for (unsigned int i = 0; i < npoints; ++i)
		localvotes[i] = Mat_<double>::zeros(dims, 1);

	// parallelise loop
#pragma omp for
	for (int currentp = 0; currentp < (int) npoints; ++currentp) {
		const Pixel &center = input.atIndex(currentp);
		const Mat_<Value> centerMat(center);

		// sort all other points by their distance to currentp (bucketsort)

		unsigned int nbuckets = PROBSHIFT_DISTBUCKETS;
		double dist2bucket = (nbuckets - 1) / (maxL2 * PROBSHIFT_MAXWIN);
		vector< vector<BsItem> > pointsBuckets(nbuckets);

		// FLANN and LSH use different index types :-(
		// -> LSH results (typically smaller) will be converted to int
		vector<int> queryIndices(maxwin);
		vector<float> queryDists;

		// old behaviour, for testing:
//		queryIndices.resize(npoints);
//		for (int i = 0; i < (int) npoints; ++i)
//			queryIndices[i] = i;

		if (config.useLSH) {
			lsh->query(currentp);
			const vector<unsigned int> &lshResult = lsh->getResult();
			queryIndices.assign(lshResult.begin(), lshResult.end());
			queryDists.clear();
			assert(lshResult.size() > 0);
			assert(queryIndices.size() > 0);
		} else {
//			queryDists.resize(maxwin);
//			queryIndices.resize(maxwin);
//			flannIndex->knnSearch(center, queryIndices, queryDists, maxwin, cv::flann::SearchParams());

			// pull corresponding row from queriesIndices
			queryIndices.assign(maxwin, -1); // <- for testing
			for (int i = 0; i < maxwin; ++i) {
				queryIndices[i] = queriesIndices(currentp, i);
			}
		}

		for (vector<int>::const_iterator it = queryIndices.begin(); it != queryIndices.end(); ++it) {
			unsigned int otherp = *it;

			// old non-lsh variant:
//		for (unsigned int otherp = 0; otherp < npoints; ++otherp) {
			const Mat_<Value> m_p(input.atIndex(otherp));
			Mat_<double> rel = m_p - centerMat;	// currentp -> otherp
			double dist = cv::norm(rel, NORM_L2);

			if (dist > maxL2 * PROBSHIFT_MAXWIN) {
				cout << "skipping distant point" << endl;
				continue;
			}

			Mat_<double> rel_norm = rel / max(DBL_EPSILON, dist);

			// append to bucket
			int bucketIdx = dist * dist2bucket;
			pointsBuckets[bucketIdx].push_back(BsItem(rel_norm, otherp, dist));
		}

		vector<double> IN_radius; // distance to each point in IN
		vector<unsigned int> IN_index; // index of each point in IN
		vector< Mat_<double> > IN_relvec; // relative vector from point in IN to center

		// determine influence neighborhood for each alpha value

		double alphas[] = {0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001}; // descending order!
//		double alphas[] = {0.005}; // for testing
		const int numalphas = sizeof(alphas) / sizeof(alphas[0]); // ugly, I know

		Mat_<double> forceVec = Mat_<double>::zeros(dims, 1);
		int alphai = 0;
		double forceMag = 0;
		int signPos = 0, signNeg = 0, forceCount = 0;
		int overlaps = 0;
		vector<double> forceMags;

		// iterate over buckets and their content
		for (unsigned int bucketIdx = 0; alphai < numalphas && bucketIdx < nbuckets; ++bucketIdx) {
			vector<BsItem>::const_iterator otherp;
			for (otherp = pointsBuckets[bucketIdx].begin();
					alphai < numalphas && otherp != pointsBuckets[bucketIdx].end();
					++otherp) {
				// add index to currentp's neighborhood (global mapping)
				infNeigh[currentp].push_back(otherp->index);

				// add currentp to list of neighborhoods otherp appears in (global mapping)
				infNeighRev[otherp->index].push_back(currentp);

				// add to local influence neighborhood
				IN_radius.push_back(otherp->radius);
				IN_index.push_back(otherp->index);
				IN_relvec.push_back(-1 * otherp->dirVec);

				// don't include overlapping points in sign test
				if (otherp->radius == 0) {
					overlaps++;
					continue;
				}

				// update force vector
				forceVec += otherp->dirVec;

				// sample force magnitude only every WINJUMP steps
				if (max(0, (int) IN_index.size() - overlaps - PROBSHIFT_WIN_SIZE) % PROBSHIFT_WINJUMP != 0) {
					// skip forward to next point
					continue;
				}

				// sample force vector magnitude
				double forceMagPrev = forceMag;
				forceMag = cv::norm(forceVec, NORM_L2);
				forceCount++;
				forceMags.push_back(forceMag);

				if (forceMagPrev < forceMag)
					signPos++;
				else if (forceMagPrev > forceMag)
					signNeg++;

				if (IN_index.size() - overlaps < PROBSHIFT_WIN_SIZE) {
					// always assume isotropy within minimum window size
					// (overlapping points don't count here)
					continue;
				}

				bool isIsotropic;
				do {
					// perform sign test for isotropy (only consider outer hull)
					isIsotropic = signtest(forceMags, alphas[alphai], PROBSHIFT_WIN_SIZE);

					if (isIsotropic)
						hasAnyIsotropy[currentp] = true;

					if (!isIsotropic) {
						// anistropy found for current alpha

						// if we're still at the minimum window size
						// and there are more alphas values to try: don't vote, try next alpha
						if (IN_index.size() - overlaps <= PROBSHIFT_WIN_SIZE
								&& alphai + 1 < numalphas) {
							alphai++;
							continue;
						}

						// compute votes (shift vectors)
						double winRadius = IN_radius.back();
//						cout << "p=" << currentp << " voting for alpha=" << alphas[alphai] << " with " << IN_index.size() << " neighbors (signPos=" << signPos << ", signNeg=" << signNeg << ")" << endl;
						for (unsigned int i = 0; i < IN_index.size(); ++i) {
							double wts = 1 - IN_radius[i] / winRadius;
							localvotes[IN_index[i]] += wts * IN_relvec[i];
							hasVote[IN_index[i]] = true;
						}

						// try next alpha
						alphai++;
					}
					// vote and try more alphas on the same window
				} while (!isIsotropic && alphai < numalphas);
			}
		}

		if (alphai < numalphas) {
			// not all alphas have been evaluated (reached maximum window size early)
			// repeat last vote once for each leftover
			tooSmallNeighborhood++;
			double factor = numalphas - alphai;
			assert(IN_radius.size() > 0);
			double winRadius = IN_radius.back();
			for (unsigned int i = 0; i < IN_index.size(); ++i) {
				double wts = (1 - IN_radius[i] / winRadius) * factor;
				localvotes[IN_index[i]] += wts * IN_relvec[i];
				hasVote[IN_index[i]] = true;
			}
		}
	}

// join votes
#pragma omp critical
	{
		for (unsigned int i = 0; i < votes.size(); ++i) {
			votes[i] += localvotes[i];
		}
	}

	// clean up
	if (lsh)
		delete lsh;
} // end omp parallel

	if (flannIndex)
		delete flannIndex;

	if (tooSmallNeighborhood)
		cout << tooSmallNeighborhood << " neighborhoods could have been larger but ran out of candidates (adjust LSH parameters or maximum window size)" << endl;

	// print some statistics of the influence neighborhoods
	{
		int nonMinimal = 0;
		for (int i = 0; i < infNeigh.size(); ++i)  {
			if (infNeigh[i].size() > PROBSHIFT_WIN_SIZE + 1)
				nonMinimal++;
		}
		cout << "Points with neighborhoods larger than absolute minimum: " << 100. * nonMinimal / npoints << "%" << endl;

		// Average size of neighborhoods a point appears in
		int nonMinimalRev = 0;
		double sizesum = 0;
		unsigned int sizecount = 0;
		for (int i = 0; i < infNeighRev.size(); ++i)  {
			// calculate average for this point
			unsigned int localsizesum = 0;
			bool appearsInNonMinimal = false;
			for (int j = 0; j < infNeighRev[i].size(); ++j)  {
				localsizesum += infNeigh[infNeighRev[i][j]].size();
				if (hasAnyIsotropy[infNeighRev[i][j]])
					appearsInNonMinimal = true;
			}
			sizesum += (double) localsizesum / infNeighRev[i].size();
			sizecount++;
			if (appearsInNonMinimal)
				nonMinimalRev++;
		}
		cout << "Points appearing in at least one neighborhood larger than absolute minimum: " << 100. * nonMinimalRev / npoints << "%" << endl;
		cout << "Average size of neighborhoods any point appears in: " << sizesum / sizecount << endl;
	}


	{ // dump neighborhood sizes as image

        vector<int> vechist;

		Mat_<float> nh(input.height, input.width);
		nh.setTo(0);
		int nhsum = 0;
		unsigned int i = 0;
		for (int y = 0; y < nh.rows; ++y) {
			for (int x = 0; x < nh.cols; ++x) {
				int s = infNeigh[i].size();
				nhsum += s;
				nh(y, x) = s;

                if (vechist.size() <= s)
                    vechist.resize(s + 1, 0);
                vechist.at(s)++;
				i++;
			}
		}

		cout << "histogram:" << endl;
        for (int i = 0; i < vechist.size(); ++i) {
            cout << i << " " << vechist[i] << endl;
        }

		double minval, maxval;
		cv::minMaxLoc(nh, &minval, &maxval);
		cerr << "Influence neighborhood sizes: avg=" << (float)nhsum/npoints << ", min=" << minval << ", max=" << maxval << endl;

	}

	int noVotes = 0;
	for (unsigned int i = 0; i < npoints; ++i) {
		if (!hasVote[i])
			noVotes++;
	}
	cout << noVotes << " points haven't got any votes" << endl;

	// construct transition probability matrix
	cout << "Constructing transition probability matrix P" << endl;

	// list of indices with non-zero transition probabilities (i.e. no outliers)
	vector<unsigned int> nooutliers;

	// list of indicies with zero transition probabilities (outliers)
	vector<unsigned int> outliers;

	// transition probability matrix as stl vectors (compressed-row storage)
	vector< vector< pair<int, double> > > crf_tpm(npoints);

	vector<double> rowsums_l1;
	vector<double> rowsums_l2;

	// for each point...
	for (unsigned int i = 0; i < npoints; ++i) {
		const Mat_<Value> iMat(input.atIndex(i));
		double rowsum_l1 = 0;
		double rowsum_l2 = 0;
		vector<unsigned int> nonzerocols;

		// if this fails, we're in trouble
		assert(infNeigh[i].size() >= PROBSHIFT_WIN_SIZE);

		// current point is an outlier as long as it has no transition probabilities to any other point
		bool isOutlier = true;

		// distance to farthest influence (for currently disabled weighting)
//		double maxDist = max(DBL_EPSILON, cv::norm(iMat, Mat_<Value>(input.atIndex(infNeigh[i].back())), NORM_L2));
		vector<unsigned int>::const_iterator infn_it;
		for (infn_it = infNeigh[i].begin(); infn_it != infNeigh[i].end(); ++infn_it) {
			unsigned int j = *infn_it;
			const Mat_<Value> jMat(input.atIndex(j));
			Mat_<double> rel = jMat - iMat;

			double dotproduct = votes[i].dot(rel);
			dotproduct /= max(DBL_EPSILON, cv::norm(votes[i], NORM_L2));
			dotproduct /= max(DBL_EPSILON, cv::norm(jMat, iMat, NORM_L2));

			// The original code applied triangular weighting here, but
			// according to the paper, it was uniform.
//			double jDist = max(DBL_EPSILON, cv::norm(jMat, iMat, NORM_L2));
//			dotproduct *= 1 - (jDist / maxDist);

			// positive half-space only
			dotproduct = max(0., dotproduct);

			if (dotproduct > 0) {
				// NB: can't fill PETSc matrix right here, because without preallocation it's really slow

				// fill crf_tpm
				crf_tpm[i].push_back(make_pair(j, dotproduct));

				nonzerocols.push_back(j);

				rowsum_l1 += dotproduct;
				rowsum_l2 += pow(dotproduct, 2);
				isOutlier = false;
			}
		}
		rowsum_l2 = sqrt(rowsum_l2);

		// prevent divisions by zero
		rowsum_l1 = max(DBL_EPSILON, rowsum_l1);
		rowsum_l2 = max(DBL_EPSILON, rowsum_l2);

		if (!isOutlier) {
			nooutliers.push_back(i);

			// record inverted rowsums for later normalization
			rowsums_l1.push_back(1. / rowsum_l1);
			rowsums_l2.push_back(1. / rowsum_l2);
		} else {
			outliers.push_back(i);
		}
	}

	cout << "done" << endl;

	cout << outliers.size() << " (" << 100. * outliers.size() / npoints << "%) points have no neighbors in the direction of their shift vector" << endl;

	// reverse lookup table for non-outliers (maps actual idx to nooutliers idx)
	unsigned int nnooutliers = nooutliers.size();
	vector<unsigned int> nooutliers_rev(npoints);
	for (unsigned int i = 0; i < nnooutliers; ++i)
		nooutliers_rev[nooutliers[i]] = i;

	int perr;
	SlepcInitialize((int *)0, (char ***) 0, (char *) 0, NULL);

	// convert stl-vectors to PETSc matrix (without normalization)
	cout << "Constructing matrix P in PETSc format..." << std::flush;

	// initialize (type and dimensions)
	PetscMat petsc_p;
    perr = MatCreate(PETSC_COMM_WORLD, &petsc_p); PETSCCHKERR(perr);
	perr = MatSetType(petsc_p, MATAIJ); PETSCCHKERR(perr);
	perr = MatSetSizes(petsc_p, nnooutliers, nnooutliers, nnooutliers, nnooutliers); PETSCCHKERR(perr);

	// fill using values from vector-based crf
	vectorCrfToPetscMat(crf_tpm, petsc_p, &nooutliers_rev);

	// construct rowsums (l1 and l2) as PETSc vector
	PetscVec petsc_rowsums_l1 = vectorToPetscVec(rowsums_l1);
	PetscVec petsc_rowsums_l2 = vectorToPetscVec(rowsums_l2);

	cout << "Assembled PETSc matrices" << endl;

	// maps each input point to a mode
	vector<unsigned int> armode(npoints, 0);

	PetscMat petsc_convp;

	if (!config.useSpectral || config.useConverged) {

		cout << "Running power iterations to compute converged P" << endl;

		// prepare PETSc matrix for converged P
		perr = MatCreate(PETSC_COMM_WORLD, &petsc_convp); PETSCCHKERR(perr);
		perr = MatSetType(petsc_convp, MATSEQAIJ); PETSCCHKERR(perr);
		perr = MatSetSizes(petsc_convp, nnooutliers, nnooutliers, nnooutliers, nnooutliers); PETSCCHKERR(perr);

		// prepare P for power iterations (L1 row-normalization)
		perr = MatDiagonalScale(petsc_p, petsc_rowsums_l1, PETSC_NULL); PETSCCHKERR(perr);

		// temporary armode array using noooutliers-indices
		vector<unsigned int> armode_no(nnooutliers, 0);

		powerIterationsPetsc(petsc_p, petsc_convp, armode_no);

		// translate to actual armode (TODO: what happens to the outliers, now?)
		for (unsigned int i = 0; i < nnooutliers; ++i)
			armode[nooutliers[i]] = nooutliers[armode_no[i]];
	}


	if (config.useSpectral) {
		// cluster identification by spectral clustering

		// intermediate armode array using noooutliers-indices
		vector<unsigned int> armode_no(nnooutliers, 0);

		if (config.useConverged) {
			// prepate converged P for spectral clustering (L2 row-normalization)

			//cout << "converged P (before normalization):" << endl;
			//MatView(petsc_convp, PETSC_VIEWER_STDOUT_SELF);

			// L2-normalize rows in converged P (that's messy, since there isn't any MatGetRowNorms() function...)
			petscL2normalize(petsc_convp);
			//cout << "converged P (after normalization):" << endl;
			//MatView(petsc_convp, PETSC_VIEWER_STDOUT_SELF);

			cout << "Using converged P for spectral clustering" << endl;

			identifySpectral(petsc_convp, armode_no);
		} else {
			// prepare P for spectral clustering
			// (L2 row-normalization using previously gathered rowsums)

			//	cout << "P (before normalization):" << endl;
			//	MatView(petsc_p, PETSC_VIEWER_STDOUT_SELF);
			//	cout << "rowsums_l2:" << endl;
			//	VecView(petsc_rowsums_l2, PETSC_VIEWER_STDOUT_SELF);
			perr = MatDiagonalScale(petsc_p, petsc_rowsums_l2, PETSC_NULL); PETSCCHKERR(perr);
			//	cout << "P (after normalization):" << endl;
			//	MatView(petsc_p, PETSC_VIEWER_STDOUT_SELF);

			cout << "Using P for spectral clustering" << endl;
//			assert(false);
			identifySpectral(petsc_p, armode_no);
		}

		// translate to actual armode (TODO: what happens to the outliers, now?)
		for (unsigned int i = 0; i < nnooutliers; ++i)
			armode[nooutliers[i]] = nooutliers[armode_no[i]];
	} else {
		// cluster identification by most probable destinations (and connected component labeling)

		// first step already happens implicitly during power iterations

		// merge connected components
		int changes;
		do {
			changes = 0;
			for (unsigned int i = 0; i < npoints; ++i) {
				for (unsigned int j = 0; j < npoints; ++j) {
					if (armode[j] == i) {
						if (armode[j] != armode[i])
							changes++;
						armode[j] = armode[i];
					}
				}
			}
			cout << "Connected components merging: " << changes << " assignments" << endl;
		} while (changes);

	}

	// cleanup
	perr = MatDestroy(petsc_p); PETSCCHKERR(perr);
	perr = SlepcFinalize(); PETSCCHKERR(perr);

    // invalid mode reference for outliers -> ignore in post processing
	for (vector<unsigned int>::iterator outlierit = outliers.begin();
            outlierit != outliers.end(); ++outlierit) {
		armode[*outlierit] = npoints + 1;
	}

	// save modes if requested
	if (!config.saveModes.empty()) {
		saveModes(armode, config.saveModes);
	}

	// Meanshift post-processing
	if (config.useMeanShift)
		armode = meanShiftPostProcess(armode, input);

	cv::Mat1s ret = assignLabels(armode, input.width, input.height);
	return ret;
}


cv::Mat1s ProbShift::assignLabels(const vector<unsigned int> &armode, int width, int height)
{
	cv::Mat1s labelImage(height, width);
	labelImage.setTo(0);
	int nextlabel = 0;
	vector<int> labels(armode.size(), -1);
	vector<int> labelsizes;
	unsigned int i = 0;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			if (armode[i] >= armode.size()) {
				// invalid reference, keep unlabeled
				++i;
				continue;
			}

			if (labels[armode[i]] == -1) {
				labels[armode[i]] = nextlabel++;
				labelsizes.push_back(0);
			}
			labelsizes[labels[armode[i]]]++;
//			std::cerr << "ret.data[" << i << "] = labels[" << armode[i] << "] = " << labels[armode[i]] << std::endl;
			labelImage(y, x) = labels[armode[i]] + 1; // add one because zero is unlabeled

			++i;
		}
	}

	{ // XXX: some statistics on label sizes (keep it? y/n)
		int lsMin = armode.size();
		int lsMax = 0;
		int lsSum = 0;
		for (unsigned int i = 0; i < labelsizes.size(); ++i) {
			lsSum += labelsizes[i];
			lsMax = max(lsMax, labelsizes[i]);
			lsMin = min(lsMin, labelsizes[i]);
		}
		cout << "Label sizes: avg=" << (lsSum / labelsizes.size()) << ", min=" << lsMin << ", max=" << lsMax << endl;
	}

	int labelcount = nextlabel;
	cout << "Assigned " << labelcount << " labels" << endl;

	if (labelcount > 255) {
		cout << "WARNING: more than 255 labels, excess will be zero (unlabeled)" << endl;
	}

	return labelImage;
}

PetscVec ProbShift::vectorToPetscVec(vector<double> &input)
{
	PetscVec output;
	int perr;
	perr = VecCreate(PETSC_COMM_WORLD, &output); PETSCCHKERR(perr);
	perr = VecSetType(output, VECSEQ); PETSCCHKERR(perr);
	perr = VecSetSizes(output, PETSC_DECIDE, input.size()); PETSCCHKERR(perr);

	// fill
	for (unsigned int i = 0; i < input.size(); ++i) {
		perr = VecSetValue(output, i, input[i], INSERT_VALUES);
	}

	// assemble
	VecAssemblyBegin(output);
	VecAssemblyEnd(output);

	return output;
}


void ProbShift::vectorCrfToPetscMat(vector< vector< pair<int, double> > > &crf, PetscMat &petsc, vector<unsigned int> *mapp)
{
	int npoints = crf.size();
	int perr;
	bool doMap = mapp != NULL;
	const vector<unsigned int> &map = mapp != NULL ? *mapp : vector<unsigned int>();

	PetscInt petsc_size;
	perr = MatGetSize(petsc, &petsc_size, PETSC_NULL); PETSCCHKERR(perr);

	// accurate preallocation based on the row vectors' size
	PetscInt nnz[petsc_size];
	memset(nnz, 0, petsc_size * sizeof(PetscInt));
	for (int crfi = 0; crfi < npoints; ++crfi) {
		unsigned int petsci = doMap ? map[crfi] : crfi;
		nnz[petsci] = crf[crfi].size();
	}
	perr = MatSeqAIJSetPreallocation(petsc, 0, nnz);
	cout << "preallocation finished..." << std::flush;

	// fill PETSc matrix row-wise using MatSetValues

	for (int crfi = 0; crfi < npoints; ++crfi) {
		unsigned int petsci = doMap ? map[crfi] : crfi;
		vector< pair<int, double> > &row = crf[crfi];

		if (row.empty())
			continue;

		// sort row by index
		std::sort(row.begin(), row.end(), CrfPred());

		// allocate arrays for values and indices
		PetscScalar *v = new PetscScalar[row.size()];
		PetscInt idxm = petsci;
		PetscInt *idxn = new PetscInt[row.size()];

		for (unsigned int n = 0; n < row.size(); ++n) {
			unsigned int petscj = doMap ? map[row[n].first] : row[n].first;
			idxn[n] = petscj;
			v[n] = row[n].second;
		}

		MatSetValues(petsc, 1, &idxm, row.size(), idxn, v, INSERT_VALUES);

		delete[] v;
		delete[] idxn;
		row.clear(); // not needed anymore
	}

	crf.clear(); // free empty row vectors

	// assemble
	MatAssemblyBegin(petsc, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(petsc, MAT_FINAL_ASSEMBLY);
	cout << "done." << endl;
}

void ProbShift::petscPreallocQuality(const PetscMat &m, double nzfact)
{
	PetscInt perr;

	MatInfo info;
	MatGetInfo(m, MAT_GLOBAL_SUM, &info);

	PetscInt w, h;
	perr = MatGetSize(m, &w, &h); PETSCCHKERR(perr);

	double finalnzfact = (double) info.nz_used / (w * h);
	double prealloc_error = (nzfact - finalnzfact) / nzfact;
	if (abs(prealloc_error) > 0.05) {
		cerr << "WARNING: PetscMat preallocation was " << 100*prealloc_error << " percent off! (prealloc=" << nzfact << ", final=" << finalnzfact << ")" << endl;
	} else {
		cerr << "YAY: preallocation was good. (error=" << prealloc_error << ", prealloc=" << nzfact << ", final=" << finalnzfact << ")" << endl;
	}
}


void ProbShift::petscL2normalize(PetscMat &m)
{
	int perr;

	PetscInt npoints;
	perr = MatGetSize(m, &npoints, PETSC_NULL); PETSCCHKERR(perr);

	// set up rowsums vector
	PetscVec rowsums_l2;
	perr = VecCreate(PETSC_COMM_WORLD, &rowsums_l2); PETSCCHKERR(perr);
	perr = VecSetType(rowsums_l2, VECSEQ); PETSCCHKERR(perr);
	perr = VecSetSizes(rowsums_l2, PETSC_DECIDE, npoints); PETSCCHKERR(perr);

	for (int row = 0; row < npoints; ++row) {
		double rowsum_l2 = 0;
		PetscInt ncols;
		const PetscInt *cols;
		const PetscScalar *vals;
		perr = MatGetRow(m, row, &ncols, &cols, &vals); PETSCCHKERR(perr);
		for (PetscInt i = 0; i < ncols; ++i)
			rowsum_l2 += pow(vals[cols[i]], 2);
		//			cout << "rowsum_l2 (before 1/sqrt): " << rowsum_l2 << endl;
		rowsum_l2 = 1./ max(sqrt(rowsum_l2), DBL_EPSILON);
		//			cout << "rowsum_l2 (after 1/sqrt): " << rowsum_l2 << endl;
		perr = VecSetValue(rowsums_l2, row, rowsum_l2, INSERT_VALUES);
		// clean up behind MatGetRow()
		perr = MatRestoreRow(m, row, &ncols, &cols, &vals); PETSCCHKERR(perr);
	}

	// assemble rowsums vector
	VecAssemblyBegin(rowsums_l2);
	VecAssemblyEnd(rowsums_l2);

	//		cout << "petsc_convp_rowsums_l2:" << endl;
	//		VecView(petsc_convp_rowsums_l2, PETSC_VIEWER_STDOUT_SELF);

	// apply normalization factors
	perr = MatDiagonalScale(m, rowsums_l2, PETSC_NULL); PETSCCHKERR(perr);

	// clean up
	VecDestroy(rowsums_l2);
}

void ProbShift::identifySpectral(PetscMat &petsc_p, vector<unsigned int> &armode, bool destroyP)
{
	int perr;

	PetscInt npoints;
	perr = MatGetSize(petsc_p, &npoints, PETSC_NULL); PETSCCHKERR(perr);

	//	cout << "used P:" << endl;
	//	MatView(petsc_usedp, PETSC_VIEWER_STDOUT_SELF);

	// construct normalized affinity matrix A = P * P'
	cout << "A = P*P'..." << flush;

	PetscMat petsc_a;
	perr = MatMatMultTranspose(petsc_p, petsc_p, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &petsc_a); PETSCCHKERR(perr);

	cout << "done." << endl;

	if (destroyP)
		perr = MatDestroy(petsc_p); PETSCCHKERR(perr);

	// set diagonal of A to zero by calling MatDiagonalSet with a vector of zeros
	PetscVec petsc_diag;
	perr = VecCreate(PETSC_COMM_WORLD, &petsc_diag); PETSCCHKERR(perr);
	perr = VecSetType(petsc_diag, VECSEQ); PETSCCHKERR(perr);
	perr = VecSetSizes(petsc_diag, PETSC_DECIDE, npoints); PETSCCHKERR(perr);
	perr = VecZeroEntries(petsc_diag); PETSCCHKERR(perr);
	perr = MatDiagonalSet(petsc_a, petsc_diag, INSERT_VALUES); PETSCCHKERR(perr);

	// reuse petsc_diag for constructing D^(-0.5)...

	// get all row sums of A
	perr = MatGetRowSum(petsc_a, petsc_diag); PETSCCHKERR(perr);

	// sqrt(1/(x+epsilon)) each element
	{
		PetscScalar    *x;
		PetscInt       i, n;
		perr = VecGetLocalSize(petsc_diag, &n); PETSCCHKERR(perr);
		VecGetArray(petsc_diag, &x);
		for(i = 0; i < n; i++) {
			x[i] = sqrt(1./(x[i] + DBL_EPSILON));
		}
		VecRestoreArray(petsc_diag, &x);
	}

	cout << "L = D^(-1/2) * A * D^(-1/2)..." << flush;

	// use MatDiagonalScale to compute laplacian
	perr = MatDiagonalScale(petsc_a, petsc_diag, petsc_diag); PETSCCHKERR(perr);

	cout << "done." << endl;

	// A is now the laplacian matrix L
	PetscMat petsc_l = petsc_a;

	// free petsc_diag
	perr = VecDestroy(petsc_diag); PETSCCHKERR(perr);

	PetscVec xr;
	PetscVec xi;
	perr = MatGetVecs(petsc_l,PETSC_NULL,&xr);PETSCCHKERR(perr);
	perr = MatGetVecs(petsc_l,PETSC_NULL,&xi);PETSCCHKERR(perr);

	// find eigenpairs using SLEPc
	EPS eps;
	const EPSType epsType;

	EPSCreate(PETSC_COMM_WORLD, &eps);
	perr = EPSSetOperators(eps,petsc_l,PETSC_NULL);PETSCCHKERR(perr);
	perr = EPSSetProblemType(eps,EPS_HEP);PETSCCHKERR(perr);
	perr = EPSSetDimensions(eps, config.maxClusts, PETSC_DECIDE, PETSC_DECIDE);PETSCCHKERR(perr);

	// this is how Matlab's eigs() does it
//	perr = EPSSetType(eps, EPSLANCZOS); PETSCCHKERR(perr);
//	perr = EPSSetTolerances(eps, DBL_EPSILON, 300); PETSCCHKERR(perr);

	perr = EPSSolve(eps);PETSCCHKERR(perr);

	/*
	   Optional: Get some information from the solver and display it
	*/
	PetscInt its, nev, maxit;
	PetscReal tol;
	perr = EPSGetIterationNumber(eps,&its);PETSCCHKERR(perr);
	perr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %d\n",its);PETSCCHKERR(perr);
	perr = EPSGetType(eps,&epsType);PETSCCHKERR(perr);
	perr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",epsType);PETSCCHKERR(perr);
	perr = EPSGetDimensions(eps,&nev,PETSC_NULL,PETSC_NULL);PETSCCHKERR(perr);
	perr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %d\n",nev);PETSCCHKERR(perr);
	perr = EPSGetTolerances(eps,&tol,&maxit);PETSCCHKERR(perr);
	perr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%d\n",tol,maxit);PETSCCHKERR(perr);

	PetscInt nconv;
	perr = EPSGetConverged(eps,&nconv);PETSCCHKERR(perr);
	perr = PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %d\n\n",nconv);PETSCCHKERR(perr);

	PetscScalar    kr, ki;

	int groups = 1 + config.maxClusts - config.minClusts;
	vector<int> group_num;
	assert(config.minClusts >= 2);
	for (int i = config.minClusts; i <= config.maxClusts; ++i) {
		group_num.push_back(i);
		cout << "group_num[] = " << i << endl;
	}
	vector< vector<int> > zp_clusters[groups]; // points per cluster per group
	double zp_quality[groups];

	if (nconv <= 0) {
		cerr << "Fatal Error: No eigenvectors found" << endl;
		exit(1);
	}

	if (nconv < config.maxClusts) {
		cerr << "Fatal Error: Not enough eigenvectors found" << endl;
		exit(1);
	}

	// extract eigenvectors from SLEPc data structures
	Mat_<double> eigenVectors(npoints, nconv); // column-wise

	cout << "eigenvectors:" << endl;
	for( PetscInt i=0; i < nconv; i++ ) {
		perr = EPSGetEigenpair(eps,i,&kr,&ki,xr,xi);PETSCCHKERR(perr);

		PetscReal re, im;
#if defined(PETSC_USE_COMPLEX)
		re = PetscRealPart(kr);
		im = PetscImaginaryPart(kr);
#else
		re = kr;
		im = ki;
#endif
		assert(im == 0);

		double *evec;
		perr = VecGetArray(xr, &evec); PETSCCHKERR(perr);
		Mat_<double> evecMat(npoints, 1, evec);
		Mat_<double> tmp = eigenVectors.col(i);
		evecMat.col(0).copyTo(tmp);
		perr = VecRestoreArray(xr, &evec); PETSCCHKERR(perr);

		// debugging prints
		cout << "eigenvalue[" << i << "] = " << kr << endl;
		/*cout << "eigenvector[" << i << "] = " << evecMat << endl;
		cout << "PetscVecView eigenvector:" << endl;
		VecView(xr, PETSC_VIEWER_STDOUT_SELF);
		cout << "eigenVectors:" << endl << eigenVectors << endl;
		assert(false);*/
	}

//	Mat_<double> Vcurr(npoints, config.maxClusts);
	Mat_<double> Vcurr(npoints, group_num[0]);
	Mat_<double> Vr;

	// prepare initial Vcurr
	Vcurr = eigenVectors.colRange(0, group_num[0]).clone();
//	for (int i = 0; i < group_num[0]; ++i) {
//		Mat_<double> tmp = Vcurr.col(i);
//		eigenVectors.col(i).copyTo(tmp);
//	}

	for(unsigned int g = 0; g < group_num.size(); ++g) {
		if (g > 0) {
			// grow Vcurr, use values from Vr
			Vcurr = Mat_<double>(npoints, group_num[g]);
			for (int i = 0; i < group_num[g] - 1; ++i) {
				Mat_<double> dest = Vcurr.col(i);
				Vr.col(i).copyTo(dest);
			}

			// add next eigenvector (use already aligned vectors)
			Mat_<double> copydest = Vcurr.col(group_num[g] - 1);
			eigenVectors.col(group_num[g] - 1).copyTo(copydest);
		}

		for (unsigned int i = 0; i < 1; ++i) {
		cout << "Rotating eigenvectors (group " << g + 1 << "/" << group_num.size() << ")..." << flush;
		// NB: colRange is [a;b), not [a;b]
		evrot(Vcurr, zp_clusters[g], zp_quality[g], Vr);
		cout << "done" << endl;
		Vcurr = Vr.clone();
	}

	}

	// determine best quality
	double maxQuality = zp_quality[0];
	double maxQualityIdx = 0;
	for (unsigned int i = 0; i < group_num.size(); ++i) {
		if (zp_quality[i] > maxQuality) {
			maxQuality = zp_quality[i];
			maxQualityIdx = i;
		}
		cout << "zp_quality[" << i << "] = " << zp_quality[i] << " (maxQ=" << maxQuality << ", maxQIdx=" << maxQualityIdx << ")" << endl;
	}

	// choose highest index with close-to-maxixum quality
	int bestIndex = maxQualityIdx;
	for (unsigned int i = group_num.size() - 1; i >= 0; --i) {
		if (abs(maxQuality - zp_quality[i]) <= 0.001) { // original paper uses 0.001
			bestIndex = i;
			break;
		}
	}

	cout << "bestIndex = " << bestIndex << endl;

/*
	for (unsigned int x = 0; x < group_num.size(); ++x) {
		cout << "dumping zp_cluster[" << x << "]..." << endl;
		for (unsigned int i = 0; i < zp_clusters[x].size(); ++i) {
			cout << "cluster " << i <<  ": (" << zp_clusters[x][i].size() << ") ";
			for (unsigned int j = 0; j < zp_clusters[x][i].size(); ++j) {
				cout << zp_clusters[x][i][j] << " ";
			}
			cout << endl;
		}
	}*/

	// use bestIndex assignments
	for (unsigned int c = 0; c < zp_clusters[bestIndex].size(); ++c) {
		vector<int>::iterator it = zp_clusters[bestIndex][c].begin();
		vector<int>::iterator itend = zp_clusters[bestIndex][c].end();
		for (; it != itend; ++it) {
			armode[*it] = c;
		}
	}

	// clean up
	cout << "destroy eps" << endl;
	perr = EPSDestroy(eps); PETSCCHKERR(perr);
	cout << "destroy petsc_a" << endl;
	perr = MatDestroy(petsc_a); PETSCCHKERR(perr);
	cout << "destroy xr" << endl;
	perr = VecDestroy(xr); PETSCCHKERR(perr);
	cout << "destroy xi" << endl;
	perr = VecDestroy(xi); PETSCCHKERR(perr);
}

void ProbShift::powerIterationsPetsc(PetscMat &p, PetscMat &convp, vector<unsigned int> &armode)
{
	bool storeConvergedP = config.useSpectral && config.useConverged;

	int perr;
	PetscInt npoints;
	perr = MatGetSize(p, &npoints, PETSC_NULL); PETSCCHKERR(perr);
	cout << "powerIterationsPetsc: npoints=" << npoints << endl;

	// TODO: might wanna use MatGetVecs here?
	PetscVec rowvec;
	perr = VecCreate(PETSC_COMM_WORLD, &rowvec); PETSCCHKERR(perr);
	perr = VecSetType(rowvec, VECSEQ); PETSCCHKERR(perr);
	perr = VecSetSizes(rowvec, PETSC_DECIDE, npoints); PETSCCHKERR(perr);

	// store converged matrix P in stl vectors (compressed-row storage)
	vector< vector< pair<int, double> > > vecconvp(npoints);

	// statistics
	int stat_itersum = 0;
	int stat_itermax = 0;
	int totalnz = 0;

	int progressInterval = npoints / 20 + 1;

	//#pragma omp parallel for PETSC is probably not thread-safe?
	for (int i = 0; i < npoints; ++i) {
		// XXX: should do that only in one thread
		if (i % progressInterval == 0) {
			cout << 100 * (i / npoints) << "%" << endl;
		}

		// get row from P
		PetscInt ncols;
		const PetscInt *cols;
		const PetscScalar *vals;
		perr = MatGetRow(p, i, &ncols, &cols, &vals); PETSCCHKERR(perr);

		if (ncols == 0)
			continue; // should never happen if outliers were excluded

		// put values into PetsVec, assemble
		perr = VecZeroEntries(rowvec); PETSCCHKERR(perr);
		perr = VecSetValues(rowvec, ncols, cols, vals, INSERT_VALUES); PETSCCHKERR(perr);
		VecAssemblyBegin(rowvec);
		VecAssemblyEnd(rowvec);

		// clean up behind MatGetRow()
		perr = MatRestoreRow(p, i, &ncols, &cols, &vals); PETSCCHKERR(perr);

		// multiply row with P until it converges
		bool converged = false;
		int iter = 0;
		double threshold = PROBSHIFT_RW_DIFFTHRESH;
		while (!converged) {

			if (iter > 500) {
				threshold = PROBSHIFT_RW_THRESHINCR * PROBSHIFT_RW_DIFFTHRESH;
			}

			if (iter > 600) {
				// give up (cycle?)
				cout << "giving up on i=" << i << endl;
				break;
			}

			PetscVec rowvec_next;
			perr = VecDuplicate(rowvec, &rowvec_next); PETSCCHKERR(perr);
			perr = VecZeroEntries(rowvec_next); PETSCCHKERR(perr);

			perr = MatMultTranspose(p, rowvec, rowvec_next);

			// measure changes using L2-norm
			perr = VecAXPY(rowvec, -1.0, rowvec_next); PETSCCHKERR(perr);
			VecAbs(rowvec);
			double norm_l2;
			{
				PetscReal tmp;
				perr = VecNorm(rowvec, NORM_2, &tmp); PETSCCHKERR(perr);
				norm_l2 = tmp;
			}
//			cout << "norm_l2 = " << norm_l2 << endl;

			converged = norm_l2 <= threshold;

			// replace rowvec with rowvec_next
			perr = VecDestroy(rowvec); PETSCCHKERR(perr);
			rowvec = rowvec_next;

			iter += 1;
		}

		stat_itersum += iter;
		stat_itermax = max(stat_itermax, iter);

		// statistics
		{
			double *rowvals;
			perr = VecGetArray(rowvec, &rowvals); PETSCCHKERR(perr);
			for (int j = 0; j < npoints; ++j) {
				if (rowvals[j] > DBL_EPSILON)
					totalnz++;
			}
			perr = VecRestoreArray(rowvec, &rowvals); PETSCCHKERR(perr);
		}

		// identify most probable destination
		PetscInt maxIdx;
		PetscReal dummyVal;
		perr = VecMax(rowvec, &maxIdx, &dummyVal); PETSCCHKERR(perr);
		std::cerr << "maxLoc[" << i << "] = " << maxIdx << " (converged after " << iter << " iterations)" << std::endl;
		armode[i] = (unsigned int) maxIdx;

		if (storeConvergedP) {
			// save row in converged P (intermediate SparseMat)
			double *rowvals;
			perr = VecGetArray(rowvec, &rowvals); PETSCCHKERR(perr);
			for (int j = 0; j < npoints; ++j) {
				double currval = rowvals[j];
				if (currval > DBL_EPSILON) {
					vecconvp[i].push_back(make_pair(j, currval));
				}
			}
			perr = VecRestoreArray(rowvec, &rowvals); PETSCCHKERR(perr);
		}

	}
	// clean up
	perr = VecDestroy(rowvec); PETSCCHKERR(perr);

	if (storeConvergedP) {
		cout << "Transferring converged rows to PETSc matrix" << endl;
		vectorCrfToPetscMat(vecconvp, convp);
	}

	// print statistics
	cout << "powerIterationsPetsc: sparsity of converged P: " << 100 * ((double) totalnz / (npoints*npoints)) << "%" << endl;
}

bool ProbShift::signtest(vector<double> &vals, double alpha, int tail) {
	int signPos = 0;
	int signNeg = 0;
	int i = 1;
	tail = tail == -1 ? vals.size() : tail;
	for (vector<double>::reverse_iterator it = vals.rbegin() + 1; i < tail && it != vals.rend(); ++it, ++i) {
		double left = *it;
		double right = *(it - 1);
		if (left < right)
			signPos++;
		if (left > right)
			signNeg++;
	}

	// perform sign test for isotropy
	double pmax = binomial_distribution<double>::find_upper_bound_on_p(signPos + signNeg, signNeg, alpha / 2);
	double pmin = binomial_distribution<double>::find_lower_bound_on_p(signPos + signNeg, signNeg, alpha / 2);
	return pmax >= 0.5 && pmin <= 0.5;
}

void ProbShift::joinModes(vector<unsigned int> &armode, const vector<int> &modes, const cv::Mat1s &labels) {
	int labelscount;
	{ // XXX: this relys on MeanShift not using label 0... not so nice!
		double tmp;
		cv::minMaxLoc(labels, NULL, &tmp);
		labelscount = tmp;
	}

	for (unsigned int l = 1; l <= labelscount; ++l) {
		// find modes belonging to this label
		vector<int> modesInLabel;
		for (unsigned int m = 0; m < modes.size(); ++m) {
			if (labels(m, 0) == l)
				modesInLabel.push_back(modes[m]);
		}

		cout << "joinModes: found " << modesInLabel.size() << " modes in label " << l << endl;

		if (modesInLabel.empty())
			continue;

		// merge connected modes
		for (unsigned int i = 0; i < armode.size(); ++i) {
			if (std::find(modesInLabel.begin(), modesInLabel.end(), armode[i]) != modesInLabel.end()) {
				// -> point i is assigned to a mode which appears in meanshift label l
				armode[i] = armode[modesInLabel[0]];
			}
		}
	}
}

vector<unsigned int> ProbShift::loadModes(std::string filename)
{
	cout << "Loading armode array from '" << filename << "'" << endl;
	vector<unsigned int> armode;
	std::ifstream armodefile(filename.c_str());

	if (!armodefile) {
		cerr << "Error: failed to open file for reading" << endl;
		return armode;
	}

	unsigned int n;
	while (armodefile >> n) {
		armode.push_back(n);
	}

	return armode;
}

void ProbShift::saveModes(const vector<unsigned int> &armode, std::string filename)
{
	cout << "Saving armode array into '" << filename << "'" << endl;
	std::ofstream armodefile(filename.c_str());
	if (!armodefile) {
		cerr << "Error: failed to open file for writing" << endl;
		return;
	}

	for (vector<unsigned int>::const_iterator it = armode.begin(); it != armode.end(); ++it) {
		armodefile << *it << ' ';
	}
}

vector<Pixel> ProbShift::getClusterCenters(const vector<int> &modes, const vector<unsigned int> &armode, const multi_img &input) {
	vector<Pixel> centersVec(modes.size());

	for (unsigned int m = 0; m < modes.size(); ++m) {
		vector<int> cpoints;
		for (int i = 0; i < armode.size(); ++i)
			if (armode[i] == modes[m])
				cpoints.push_back(i);

		cv::Mat_<Value> cluster(input.size(), cpoints.size());
		for (int i = 0; i < cpoints.size(); ++i) {
			cv::Mat_<Value> p(input.atIndex(cpoints[i]));
			cv::Mat_<Value> tmp = cluster.col(i);
			p.copyTo(tmp);
		}

		cv::Mat_<Value> avg;
		cv::reduce(cluster, avg, 1, CV_REDUCE_AVG);

		centersVec[m].resize(input.size());
		for (int i = 0; i < input.size(); ++i)
			centersVec[m][i] = avg(i, 0);
	}

	return centersVec;
}

vector<unsigned int> ProbShift::meanShiftPostProcess(const vector<unsigned int> &armode, const multi_img &input) {
	// build list of modes
	vector<bool> isMode(armode.size(), false);
	for (int i = 0; i < armode.size(); ++i) {
		if (armode[i] <= isMode.size()) // disregard invalid references (could be marked outliers)
			isMode.at(armode[i]) = true;
	}
	vector<int> modes;
	for (int i = 0; i < isMode.size(); ++i) {
		if (isMode[i])
			modes.push_back(i);
	}

	// find geometric centers of the clusters
	vector<Pixel> centersVec = getClusterCenters(modes, armode, input);

	// create multi_img from clustercenters
	multi_img centersMI(modes.size(), 1, input.size());
	centersMI.minval = input.minval;
	centersMI.maxval = input.maxval;
	for (unsigned int m = 0; m < modes.size(); ++m) {
		// don't use the matrix version of multi_img::setPixel (uses broken OpenCV iterator)!
		centersMI.setPixel(m, 0, centersVec[m]);
	}
	centersMI.rebuildPixels();

	// determine the variance, stddev and range of each cluster, for use as bandwidth value in meanshift

	vector<double> variances(modes.size(), 0);
	vector<double> stddevs(modes.size(), 0);
	vector<double> ranges(modes.size(), 0);

	for (unsigned int m = 0; m < modes.size(); ++m) {
		vector<int> cpoints;
		for (int i = 0; i < armode.size(); ++i)
			if (armode[i] == modes[m])
				cpoints.push_back(i);

		Pixel &center = centersVec[m];

		for (int i = 0; i < cpoints.size(); ++i) {
			const Pixel &p = input.atIndex(cpoints[i]);
			double dist = cv::norm(Mat_<Value>(center), Mat_<Value>(p), NORM_L1);
			variances[m] += pow(dist, 2);
			if (dist > ranges[m])
				ranges[m] = dist;
		}
		variances[m] /= cpoints.size();
		stddevs[m] = sqrt(variances[m]);
		cout << "variance[" << m << "]: " << variances[m] << ", stddev=" << stddevs[m] << ", range=" << ranges[m] << "(points: " << cpoints.size() << ")" << endl;
	}

	// run Meanshift

	int expectedmodes = config.msClusts;
	MeanShiftConfig msconf;
	msconf.k = 1;
	msconf.pruneMinN = 1;
	msconf.batch = true;

	MeanShift ms(msconf);

	cv::Mat1s mslabels;
	Labeling labeling;
	labeling.yellowcursor = false;
	vector<double> bandwidths(modes.size(), 0);
    double bwfactor = config.msBwFactor;

	if (config.msClusts <= 0) {
		for (int m = 0; m < modes.size(); ++m) {
			bandwidths[m] = stddevs[m] * bwfactor;
		}
		mslabels = ms.execute(centersMI, NULL, &bandwidths);
	} else {
		// try to find a bwfactor yielding the requested number of clusters

		int maxiter = PROBSHIFT_MSPOSTPROC_MAXITER;
		int foundmodes = 0;

		double searchstep = 1;
		while (maxiter-- > 0) {
			for (int m = 0; m < modes.size(); ++m) {
				bandwidths[m] = stddevs[m] * bwfactor;
			}
			mslabels = ms.execute(centersMI, NULL, &bandwidths);

			labeling = mslabels;
			foundmodes = labeling.getCount() - 1;

			if ((foundmodes > expectedmodes && searchstep < 0)
				|| (foundmodes < expectedmodes && searchstep > 0))
				searchstep = -0.5 * searchstep; // reverse direction, decrease step size
			else
				searchstep *= 1.5; // keep direction, increase step size

			if (foundmodes == expectedmodes) {
				cout << "Match: bwfactor = " << bwfactor << endl;
				break;
			}

			// next bwfactor (make sure it's never zero)
			bwfactor = max(DBL_EPSILON, bwfactor + searchstep);

			cout << "binary search: got " << foundmodes << " modes (want " << expectedmodes << "), updated bwfactor by " << searchstep << ": " << bwfactor << endl;
		}
		if (maxiter <= 0) {
			cerr << "Error: reached maximum iterations limit (continuing with merging any modes)" << endl;
			return armode;
		}
	}

	// join our modes based on the meanshift clustering
	vector<unsigned int> armodecopy = armode;
	joinModes(armodecopy, modes, mslabels);

	return armodecopy;
}

bool ProbShift::progressUpdate(int percent)
{
	if (progressObserver == NULL)
		return true;

	return progressObserver->update(percent);
}



} // namespace
