/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "medianshift.h"
#include <lsh.h>
#include <lshreader.h>

#include <multi_img.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <queue>

namespace vole {

using std::cerr;
using std::cout;
using std::endl;
using std::setprecision;

void MedianShift::removeDups(vector<unsigned int> &v) {
	std::sort(v.begin(), v.end());
	v.erase(unique(v.begin(), v.end()), v.end());
}

vector<unsigned int> MedianShift::pruneByL2(const multi_img &image, const vector<unsigned int> &points, unsigned int center, double radius) {
	const vector<Value> &c = image.atIndex(center);
	vector<unsigned> ret;
	int removed = 0;
	for (vector<unsigned int>::const_iterator it = points.begin(); it != points.end(); ++it) {
		double dist;
		if (distL2(c, image.atIndex(*it), radius, dist)) {
			ret.push_back(*it);
		} else {
			removed++;
		}
	}
	return ret;
}

vector<unsigned int> MedianShift::pruneByL1(const multi_img &image, const vector<unsigned int> &points, unsigned int center, double radius) {
	const vector<Value> &c = image.atIndex(center);
	vector<unsigned> ret;
	int removed = 0;
	for (vector<unsigned int>::const_iterator it = points.begin(); it != points.end(); ++it) {
		double dist;
		if (distL1(c, image.atIndex(*it), radius, dist)) {
			ret.push_back(*it);
		} else {
			removed++;
		}
	}
	return ret;
}


unsigned int MedianShift::getTukeyMedian(const multi_img &img, const vector<unsigned int> &points, int nprojections, const vector<unsigned int> &weights)
{
	int npoints = points.size();
	/// put points in dim*n matrix
	Mat_<Value> pointsMat(img.size(), npoints);
	/// map lines in pointsMat to indices in whole image
	std::vector<unsigned int> indices(npoints);
	int i = 0;

	vector<unsigned int>::const_iterator points_it;
	for (points_it = points.begin(); points_it < points.end(); ++points_it) {
		Mat_<Value> bucketcol = pointsMat.col(i);
		Mat_<Value>(img.atIndex(*points_it)).copyTo(bucketcol);
		indices[i] = *points_it;
		i++;
	}

	unsigned int median = getTukeyMedian(pointsMat, nprojections, weights);

	return indices[median];
}

unsigned int MedianShift::getTukeyMedian(const Mat_<Value> &points, int nprojections, const vector<unsigned int> &weights)
{
	unsigned int npoints = points.cols;
	bool weighted = !weights.empty();

	/// generate random projections
	Mat_<Value> projVecs(nprojections, points.rows); /// nprojections*dims
	cv::randu(projVecs, 0, 1);

	/// apply random projections
	Mat_<Value> proj = projVecs * points;

	/// find Tukey depth for each column (=point)

	Mat_<Value> proj_order(nprojections, npoints);
	Mat_<int> proj_tmp(1, npoints);

	/// for each projection...
	for (int p = 0; p < nprojections; ++p) {
		/// get row in projections matrix
		Mat_<Value> row = proj.row(p);

		/// get row in order matrix
		Mat_<Value> irow = proj_order.row(p);

		/// fill temp matrix with ascending values (XXX: where's cvRange() in opencv-2.2?!)
		for (unsigned int i = 0; i < npoints; ++i) { proj_tmp(0, i) = i; }

		/// sort temp matrix by comparing projection values
		std::sort(proj_tmp.begin(), proj_tmp.end(), TukeyPred(row));

		/// fill current row of proj_order with the values' sorted positions
		unsigned int w = 0;
		for (unsigned int i = 0; i < npoints; ++i) {
			if (weighted) {
				w += weights[proj_tmp(0, i)];
			} else {
				w = i;
			}
			proj_order(p, proj_tmp(0, i)) = (float)w;
		}
	}

	/// "flip" greater half of values (0..n -> 0..n/2..0)
	Mat_<Value> proj_order_inv(nprojections, npoints);
	if (weighted) {
		Scalar weightsum = cv::sum(Mat(weights));
		cv::subtract(weightsum, proj_order, proj_order_inv);
	} else {
		cv::subtract(cv::Scalar(npoints - 1), proj_order, proj_order_inv);
	}
	proj_order = cv::min(proj_order_inv, proj_order);

	/// now, the minimum of each column in proj_order contains the Tukey depth for that point

	Mat_<Value> tdepths(1, npoints);
	cv::reduce(proj_order, tdepths, 0, CV_REDUCE_MIN);

	double maxDepthVal;
	cv::Point maxDepthLoc;
	cv::minMaxLoc(tdepths, NULL, &maxDepthVal, NULL, &maxDepthLoc);

	return maxDepthLoc.x;
}

cv::Mat1s MedianShift::execute(const multi_img& input, ProgressObserver *progress) {
	cout << "Median Shift Segmentation" << endl;

	progressObserver = progress;

	/// XXX: get rid of that?
	const multi_img *inputp = &input;

	/// input properties
	unsigned int npoints = inputp->width * inputp->height;
	unsigned int dims = inputp->size();
	multi_img::Range range = input.data_range();
	double maxL1 = (range.max - range.min) * dims;

	/// TODO: make LSH feed on multi_img directly?
	data = inputp->export_interleaved(true);

	// TODO: configurable fixed seed
//	cout << "using fixed seed 42" << endl;
//	srand(42);

	LSH lsh_data(data, npoints, inputp->size(), config.K, config.L, false);
	LSHReader lsh(lsh_data);
	LSH lsh_ddp_data(data, npoints, inputp->size(), config.K, config.L, true);
	LSHReader lsh_ddp(lsh_ddp_data);

	cout << "Computing adaptive window sizes using k=" << config.k << endl;

	/// make sure pixel cache is up to date
	inputp->rebuildPixels();

	/// adaptive window size for each point
	vector<double> windowSizes(npoints);
	unsigned int pilotk = config.k; // TODO: why is config.k of type double?

	/// statistics/debugging
	double dbg_winavg = 0;
	unsigned int dbg_noknn = 0;

	unsigned int nbuckets = MEDIANSHIFT_DISTBUCKETS;
	double dist2bucket = (nbuckets - 1) / (maxL1 * MEDIANSHIFT_MAXWIN);
	for (unsigned int currentp = 0; currentp < npoints; ++currentp) {
		const multi_img::Pixel &center = inputp->atIndex(currentp);

		/// sort points in neighborhood by their distance to currentp (bucketsort)

		vector<unsigned int> buckets(nbuckets, 0);

		lsh_ddp.query(currentp);
		const vector<unsigned int> lshResult = lsh_ddp.getResult();
		for (vector<unsigned int>::const_iterator it = lshResult.begin(); it != lshResult.end(); ++it) {
			double dist;
			distL1(center, inputp->atIndex(*it), 0, dist);

			if (dist <= maxL1 * MEDIANSHIFT_MAXWIN) {
				buckets.at((int) (dist * dist2bucket))++;
			}
		}

		/// determine distance to k-nearest neighbor
		unsigned int knn;
		unsigned int sum = 0;
		for (knn = 0; knn < nbuckets && sum < pilotk; ++knn) {
			sum += buckets[knn];
		}

		if (sum < pilotk) {
			/// less than k points within maximum window
			dbg_noknn++;
		}

		windowSizes[currentp] = (double) (knn + 1) / dist2bucket;
		dbg_winavg += windowSizes[currentp];
	}

	dbg_winavg /= npoints;
	cout << "Average window size: " << dbg_winavg << endl;
	cout << "No kNN found for " << setprecision(3) << (double) dbg_noknn / npoints * 100  << "% of all points" << endl;

	/// indices of representative points
	vector<unsigned int> repPoints;

	if (config.signifThresh >= 0) {
		cout << "Significant mode detection: starting with medians of largest LSH buckets (threshold " << config.signifThresh << ")" << endl;

		vector< vector<unsigned int> > largest = lsh.lsh.getLargestBuckets(config.signifThresh);

		vector< vector<unsigned int> >::const_iterator buckets_it;
		for (buckets_it = largest.begin(); buckets_it < largest.end(); ++buckets_it) {
			unsigned int median = getTukeyMedian(*inputp, *buckets_it, MEDIANSHIFT_TUKEY_NPROJ);
			repPoints.push_back(median);
		}

		/// keep only unique representatives
		removeDups(repPoints);

		cout << "Found " << repPoints.size() << " unique representative points in " << largest.size() << " buckets" << endl;
	} else {
		cout << "Using all points as initial representatives" << endl;

		for (unsigned int i = 0; i < npoints; ++i)
			repPoints.push_back(i);
	}

	progressUpdate(5);

	/// connect reps with modes in single shift step, determine weights
	vector<unsigned int> repWeights(repPoints.size(), 0);
	vector<unsigned int> armode(npoints);
	vector<bool> armode_valid(npoints, false);
	for (unsigned int i = 0; i < repPoints.size(); ++i) {
		int p = repPoints[i];
		lsh.query(p);

		///const vector<unsigned int> nn = pruneByL2(*inputp, lsh.getResult(), p, config.radius);
		const vector<unsigned int> nn = pruneByL1(*inputp, lsh.getResult(), p, windowSizes[p]);

		repWeights[i] = nn.size();
		assert(nn.size() != 0);

		unsigned int median = getTukeyMedian(*inputp, nn, MEDIANSHIFT_TUKEY_NPROJ);
		armode[p] = median;
		armode_valid[p] = true;
	}

	// refine modes by repeatedly replacing them with the median of their local neighborhood
	for (unsigned int i = 0; i < repPoints.size(); ++i) {
		int p = repPoints[i];
		if (!armode_valid[p])
			continue; /// ignore

		for (int loop = 0; loop < 20 && (armode[p] != armode[armode[p]] || !armode_valid[armode[p]]); ++loop) {
			if (!armode_valid[armode[p]]) {
				/// unresolved point -> find median of neighborhood
				lsh.query(armode[p]);
				///const vector<unsigned int> nn = pruneByL2(*inputp, lsh.getResult(), armode[p], radius);
				const vector<unsigned int> nn = pruneByL1(*inputp, lsh.getResult(), armode[p], windowSizes[armode[p]]);
				unsigned int median = getTukeyMedian(*inputp, nn, MEDIANSHIFT_TUKEY_NPROJ);
				armode[armode[p]] = median;
				armode_valid[armode[p]] = true;
			}
			armode[p] = armode[armode[p]];
		}
	}

	/// acquire list of unique modes
	vector<unsigned int> modes;
	for (unsigned int i = 0; i < repPoints.size(); ++i) {
		if (armode_valid[repPoints[i]]) {
			modes.push_back(armode[repPoints[i]]);
			std::cerr << "adding mode for repPoint " << repPoints[i] << " with weight=" << repWeights[i] << std::endl;
		}
	}

	/// keep only unique modes
	removeDups(modes);

	/// propagate sums of weights to modes
	vector<unsigned int> mode_weights(modes.size(), 0);
	for (unsigned int m = 0; m < modes.size(); ++m) {
		for (unsigned int r = 0; r < repPoints.size(); ++r) {
			assert(armode_valid[repPoints[r]]); /// <- not sure if this is always true... let's see...
			if (armode_valid[repPoints[r]] && armode[repPoints[r]] == modes[m]) {
				mode_weights[m] += repWeights[r];
			}
		}
	}

	for (unsigned int m = 0; m < modes.size(); ++m) {
		std::cerr << "Mode " << m << " at index " << modes[m] << " has weight of " << mode_weights[m] << std::endl;
	}

	progressUpdate(10);
	std::cerr << "Starting iterations on " << modes.size() << " modes" << std::endl;

	/// iteratively shift modes until convergence (only operate on set of modes instead of whole dataset)
	/// XXX: use adaptive window size here?
	vector<unsigned int> *prevModes = new vector<unsigned int>(modes);
	vector<unsigned int> *prevWeights = new vector<unsigned int>(mode_weights);

	double radiusFactor = 1;
	for (int iterc = 0; iterc < MEDIANSHIFT_MAXITER; ++iterc) {
		LSH lsh_modes_data = LSH(data, npoints, inputp->size(), config.K, config.L, false, *prevModes);
		LSHReader lsh_modes(lsh_modes_data);

		vector<unsigned int> *newModes = new vector<unsigned int>();
		vector<unsigned int> *newWeights = new vector<unsigned int>();

		bool modesChanged = false;
		for (unsigned int i = 0; i < prevModes->size(); ++i) {
			unsigned int p = (*prevModes)[i];
			lsh_modes.query(p);
//			const vector<unsigned int> nn = pruneByL2(*inputp, lsh_modes.getResult(), p, config.radius);
			const vector<unsigned int> nn = pruneByL1(*inputp, lsh_modes.getResult(), p, windowSizes[p] * radiusFactor);
			std::cerr << "iterations: lsh query (pruned) returned " << nn.size() << " points" << std::endl;

			assert(nn.size() != 0);

			unsigned int median = getTukeyMedian(*inputp, nn, MEDIANSHIFT_TUKEY_NPROJ, *prevWeights);
			newModes->push_back(median);

			armode[p] = median;
			armode_valid[p] = true;

			if (p != median) {
				modesChanged = true;
			}
		}

		if (modesChanged) {

			/// resolve chain-linkings in armode
			for (unsigned int i = 0; i < prevModes->size(); ++i) {
				int p = (*prevModes)[i];
				for (int loop = 0; loop < 20 && (armode[p] != armode[armode[p]]); ++loop) {
					armode[p] = armode[armode[p]];
					assert(armode_valid[armode[p]]);
				}
			}

			/// put leftovers in list of new modes
			for (unsigned int i = 0; i < prevModes->size(); ++i) {
				int p = (*prevModes)[i];
				newModes->push_back(armode[p]);
			}

			/// keep only unique modes
			removeDups(*newModes);

			/// propagate sums of weights to new modes
			newWeights->resize(newModes->size(), 0);
			for (unsigned int m = 0; m < newModes->size(); ++m) {
				for (unsigned int pm = 0; pm < prevModes->size(); ++pm) {
					unsigned int pm_p = (*prevModes)[pm];
					if (armode_valid[pm_p] && armode[pm_p] == (*newModes)[m]) {
						(*newWeights)[m] += (*prevWeights)[pm];
					}
				}
			}

			for (unsigned int m = 0; m < newModes->size(); ++m) {
				std::cerr << "New mode " << m << " at index " << newModes->at(m) << " has weight of " << newWeights->at(m) << std::endl;
			}
		}



		if (!modesChanged || (prevModes->size() == newModes->size() && *prevModes == *newModes)) {
			std::cerr << "Breaking after " << iterc+1 << " iterations" << std::endl;
			if (!modesChanged)
				std::cerr << "(Final iteration didn't yield any new medians)" << std::endl;
			iterc = MEDIANSHIFT_MAXITER;
		}

		if (modesChanged) {
			delete prevModes;
			delete prevWeights;

			prevModes = newModes;
			prevWeights = newWeights;
		}

		radiusFactor += .1;
	}

	progressUpdate(15);
	std::cerr << "Have " << prevModes->size() << " points left after convergence" << std::endl;

	for (unsigned int m = 0; m < prevModes->size(); ++m) {
		std::cerr << "Final mode " << m << " at index " << prevModes->at(m) << " (weight=" << prevWeights->at(m) << ")" << std::endl;
	}

	/// prepare output mask
	cv::Mat1s ret(inputp->height, inputp->width);
	ret.setTo(0); /// 0 means unlabeled, first mode is 1

	if (config.skipprop) {
		for (unsigned int m = 0; m < prevModes->size(); ++m) {
			unsigned int i = (*prevModes)[m];
			ret(i - (i % ret.cols), i % ret.cols) = m + 1;
		}
	} else {
		LSH lsh_new_data(data, npoints, inputp->size(), config.K, config.L, true);
		LSHReader lsh_new(lsh_new_data);
		vector<unsigned int> assigned = propagateModes(lsh_new, *prevModes, *inputp, windowSizes);

		unsigned int i = 0;
		for (cv::Mat1s::iterator it = ret.begin(); it != ret.end(); ++it) {
			*it = assigned[i] + 1;
			++i;
		}
	}

	delete prevModes;
	delete prevWeights;

	return ret;
}

vector<unsigned int> MedianShift::propagateModes(LSHReader &lsh, const vector<unsigned int> &modes, const multi_img &img, const vector<double> &windowSizes) {
	const unsigned int npoints = img.width * img.height;
	std::priority_queue<GpItem> queue;

	vector<bool> closed(npoints, false);
	unsigned int closedCounter = 0;
	vector<unsigned int> ret(npoints);

	/// precompute all neighborhoods
	std::cerr << "Precomputing neighborhoods" << std::endl;
	vector< vector<unsigned int> > nh;
	for (unsigned int i = 0; i < npoints; ++i) {
		lsh.query(i);
		///nh.push_back(pruneByL2(img, lsh.getResult(), i, config.radius));
		nh.push_back(pruneByL1(img, lsh.getResult(), i, windowSizes[i]));
	}

	/// put all modes in queue
	for (unsigned int i = 0; i < modes.size(); ++i) {
		queue.push(GpItem(0, modes[i], i));
	}

	std::cerr << "Propagating modes" << std::endl;
	while(!queue.empty()) {
		const GpItem current = queue.top();
		queue.pop();
		if (closed[current.index]) {
			continue; /// skip
		}

		/// mark as closed, assign to mode
		closed[current.index] = true;
		closedCounter++;
		ret[current.index] = current.mode;

		/// put all non-closed neighbors in queue
		vector<unsigned int> &nn = nh[current.index];
		for (unsigned int j = 0; j < nn.size(); ++j) {
			unsigned int nidx = nn[j];

			if (closed[nidx])
				continue;

			double d;
			distL2(img.atIndex(nidx), img.atIndex(current.index), 0, d);
			queue.push(GpItem(d + current.dist, nidx, current.mode));
		}

		/// update progress
		if (closedCounter % (npoints/20) == 0) {
			progressUpdate(15 + 85 * closedCounter/npoints);
		}
	}

	return ret;
}

bool MedianShift::progressUpdate(int percent)
{
	if (progressObserver == NULL)
		return true;

	return progressObserver->update(percent);
}


} // namespace
