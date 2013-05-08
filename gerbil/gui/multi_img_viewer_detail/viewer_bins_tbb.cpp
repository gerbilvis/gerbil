
#include "multi_img.h"

#include <tbb/blocked_range2d.h>
#include <tbb/task.h>
#include <algorithm>

#include <background_task.h>
#include "../viewport.h"

#include "viewer_bins_tbb.h"
#include "curpos.h"

#define REUSE_THRESHOLD 0.1


class Accumulate {
public:
	Accumulate(bool substract, multi_img &multi, cv::Mat1s &labels, cv::Mat1b &mask,
		int nbins, multi_img::Value binsize, multi_img::Value minval, bool ignoreLabels,
		std::vector<multi_img::Value> &illuminant,
		std::vector<BinSet> &sets)
		: substract(substract), multi(multi), labels(labels), mask(mask), nbins(nbins), binsize(binsize),
		minval(minval), illuminant(illuminant), ignoreLabels(ignoreLabels), sets(sets) {}
	void operator()(const tbb::blocked_range2d<int> &r) const;
private:
	bool substract;
	multi_img &multi;
	cv::Mat1s &labels;
	cv::Mat1b &mask;
	int nbins;
	multi_img::Value binsize;
	multi_img::Value minval;
	bool ignoreLabels;
	std::vector<multi_img::Value> &illuminant;
	std::vector<BinSet> &sets;
};

bool ViewerBinsTbb::run()
{
	bool reuse = ((!add.empty() || !sub.empty()) && !inplace);
	bool keepOldContext = false;
	if (reuse) {
		keepOldContext = ((fabs(args.minval) * REUSE_THRESHOLD) >= (fabs(args.minval - (*multi)->minval))) &&
			((fabs(args.maxval) * REUSE_THRESHOLD) >= (fabs(args.maxval - (*multi)->maxval)));
		if (!keepOldContext) {
			reuse = false;
			add.clear();
			sub.clear();
		}
	}

	std::vector<BinSet> *result;
	SharedDataSwapLock current_lock(current->mutex, boost::defer_lock_t());
	if (reuse) {
		SharedDataSwapLock temp_wlock(temp->mutex);
		assert(multi);
		result = temp->swap(NULL);
		if (!result) {
			result = new std::vector<BinSet>(**current);
		} else {
			for (int i = result->size(); i < colors.size(); ++i) {
				result->push_back(BinSet(colors[i], (*multi)->size()));
			}
		}
	} else if (inplace) {
		current_lock.lock();
		assert(multi);
		result = &(**current);
		for (int i = result->size(); i < colors.size(); ++i) {
			result->push_back(BinSet(colors[i], (*multi)->size()));
		}
	} else {
		result = new std::vector<BinSet>();
		assert(multi);
		for (int i = 0; i < colors.size(); ++i) {
			result->push_back(BinSet(colors[i], (*multi)->size()));
		}
		add.push_back(cv::Rect(0, 0, (*multi)->width, (*multi)->height));
	}

	if (!args.dimensionalityValid) {
		if (!keepOldContext)
			args.dimensionality = (*multi)->size();
		args.dimensionalityValid = true;
	}

	if (!args.metaValid) {
		if (!keepOldContext)
			args.meta = (*multi)->meta;
		args.metaValid = true;
	}

	if (!args.labelsValid) {
		if (!keepOldContext) {
			args.labels.resize((*multi)->size());
			for (unsigned int i = 0; i < (*multi)->size(); ++i) {
				if (!(*multi)->meta[i].empty)
					args.labels[i].setNum((*multi)->meta[i].center);
			}
		}
		args.labelsValid = true;
	}

	if (!args.binsizeValid) {
		if (!keepOldContext)
			args.binsize = ((*multi)->maxval - (*multi)->minval) / (multi_img::Value)(args.nbins - 1);
		args.binsizeValid = true;
	}

	if (!args.minvalValid) {
		if (!keepOldContext)
			args.minval = (*multi)->minval;
		args.minvalValid = true;
	}

	if (!args.maxvalValid) {
		if (!keepOldContext)
			args.maxval = (*multi)->maxval;
		args.maxvalValid = true;
	}

	std::vector<cv::Rect>::iterator it;
	for (it = sub.begin(); it != sub.end(); ++it) {
		Accumulate substract(true, **multi, labels, mask, args.nbins, args.binsize, args.minval, args.ignoreLabels, illuminant, *result);
		tbb::parallel_for(
			tbb::blocked_range2d<int>(it->y, it->y + it->height, it->x, it->x + it->width),
				substract, tbb::auto_partitioner(), stopper);
	}
	for (it = add.begin(); it != add.end(); ++it) {
		Accumulate add(
			false, **multi, labels, mask, args.nbins, args.binsize, args.minval, args.ignoreLabels, illuminant, *result);
		tbb::parallel_for(
			tbb::blocked_range2d<int>(it->y, it->y + it->height, it->x, it->x + it->width),
				add, tbb::auto_partitioner(), stopper);
	}

	if (stopper.is_group_execution_cancelled()) {
		delete result;
		return false;
	} else {
		if (reuse && !apply) {
			SharedDataSwapLock temp_wlock(temp->mutex);
			temp->swap(result);
		} else if (inplace) {
			current_lock.unlock();
		} else {
			SharedDataSwapLock context_wlock(context->mutex);
			SharedDataSwapLock current_wlock(current->mutex);
			**context = args;
			delete current->swap(result);
		}
		return true;
	}
}

void Accumulate::operator()(const tbb::blocked_range2d<int> &r) const
{
	for (int y = r.rows().begin(); y != r.rows().end(); ++y) {
		short *lr = labels[y];
		for (int x = r.cols().begin(); x != r.cols().end(); ++x) {
			if (!mask.empty() && !mask(y, x))
				continue;

			int label = (ignoreLabels ? 0 : lr[x]);
			label = (label >= sets.size()) ? 0 : label;
			const multi_img::Pixel& pixel = multi(y, x);
			BinSet &s = sets[label];

			BinSet::HashKey hashkey(boost::extents[multi.size()]);
			for (int d = 0; d < multi.size(); ++d) {
				int pos = floor(
					curpos(pixel[d], d, minval, binsize, illuminant));
				pos = std::max(pos, 0); pos = std::min(pos, nbins-1);
				hashkey[d] = (unsigned char)pos;
			}

			if (substract) {
				BinSet::HashMap::accessor ac;
				if (s.bins.find(ac, hashkey)) {
					ac->second.sub(pixel);
					if (ac->second.weight == 0.f)
						s.bins.erase(ac);
				}
				ac.release();
				s.totalweight--; // atomic
			} else {
				BinSet::HashMap::accessor ac;
				s.bins.insert(ac, hashkey);
				ac->second.add(pixel);
				ac.release();
				s.totalweight++; // atomic
			}
		}
	}
}
