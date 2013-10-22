#include "distviewbinstbb.h"

#include <background_task/background_task.h>
#include <multi_img.h>

#include <algorithm>
#include <tbb/partitioner.h>
#include <tbb/parallel_for.h>

#include <gerbil_gui_debug.h>

#define REUSE_THRESHOLD 0.1


class Accumulate {
public:
	Accumulate(bool subtract, multi_img &multi, const cv::Mat1s &labels, const cv::Mat1b &mask,
		int nbins, multi_img::Value binsize, multi_img::Value minval, bool ignoreLabels,
		std::vector<multi_img::Value> &illuminant,
		std::vector<BinSet> &sets)
		: subtract(subtract), multi(multi), labels(labels), mask(mask), nbins(nbins), binsize(binsize),
		minval(minval), illuminant(illuminant), ignoreLabels(ignoreLabels), sets(sets) {}
	void operator()(const tbb::blocked_range2d<int> &r) const;
private:
	bool subtract;
	multi_img &multi;
	const cv::Mat1s &labels;
	const cv::Mat1b &mask;
	int nbins;
	multi_img::Value binsize;
	multi_img::Value minval;
	bool ignoreLabels;
	std::vector<multi_img::Value> &illuminant;
	std::vector<BinSet> &sets;
};

#include <opencv2/highgui/highgui.hpp>

bool DistviewBinsTbb::run()
{
	//cv::imwrite("/tmp/mask.png", mask);
	bool reuse = ((!add.empty() || !sub.empty()) && !inplace);
	bool keepOldContext = false;
	if (reuse) {
		/* test if minval / maxval differ too much to re-use data */
		keepOldContext = ((fabs(args.minval) * REUSE_THRESHOLD) >=
						  (fabs(args.minval - (*multi)->minval))) &&
			((fabs(args.maxval) * REUSE_THRESHOLD) >=
			 (fabs(args.maxval - (*multi)->maxval)));
		if (!keepOldContext) {
			reuse = false;
			add.clear();
			sub.clear();
		}
	}
	std::vector<BinSet> *result;
	SharedDataSwapLock current_lock(current->mutex, boost::defer_lock_t());
	if (reuse) {
		// TODO: we still operate on the data afterwards (we are the only ones
		// that work on it), this needs to be done properly, maybe by just not
		// using shared data for temp as it is not really shared
		SharedDataSwapLock temp_wlock(temp->mutex);
		assert(multi);
		result = &(**temp);
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
			args.xlabels.resize((*multi)->size());
			for (unsigned int i = 0; i < (*multi)->size(); ++i) {
				if (!(*multi)->meta[i].empty) {
					args.xlabels[i].setNum((*multi)->meta[i].center);
				} else {
					//GGDBGM(i << " meta is empty. "<< args.type << endl);
				}
			}
		}
		args.labelsValid = true;
	}

	if (!args.binsizeValid) {
		if (!keepOldContext)
			args.binsize = ((*multi)->maxval - (*multi)->minval)
							/ (multi_img::Value)(args.nbins - 1);
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
		Accumulate substract(true, **multi, labels, mask, args.nbins,
							 args.binsize, args.minval, args.ignoreLabels,
							 illuminant, *result);
		tbb::parallel_for(
			tbb::blocked_range2d<int>(it->y, it->y + it->height,
									  it->x, it->x + it->width),
				substract, tbb::auto_partitioner(), stopper);
	}
	for (it = add.begin(); it != add.end(); ++it) {
		Accumulate add(
			false, **multi, labels, mask, args.nbins, args.binsize,
					args.minval, args.ignoreLabels, illuminant, *result);
		tbb::parallel_for(
			tbb::blocked_range2d<int>(it->y, it->y + it->height,
									  it->x, it->x + it->width),
				add, tbb::auto_partitioner(), stopper);
	}

	if (stopper.is_group_execution_cancelled()) {
		delete result;
		return false;
	} else {
		if (reuse && !apply) {
			SharedDataSwapLock temp_wlock(temp->mutex);
			std::cerr << "TODO REPLACE" << std::endl;
			//temp->replace(result);
		} else if (inplace) {
			current_lock.unlock();
		} else {
			SharedDataSwapLock context_wlock(context->mutex);
			SharedDataSwapLock current_wlock(current->mutex);
			**context = args;
			current->replace(result);
		}
		return true;
	}
}

void Accumulate::operator()(const tbb::blocked_range2d<int> &r) const
{
	for (int y = r.rows().begin(); y != r.rows().end(); ++y) {
		const short *lr = labels[y];
		const uchar *mr = (mask.empty() ? 0 : mask[y]);
		for (int x = r.cols().begin(); x != r.cols().end(); ++x) {
			if (mr && !mr[x])
				continue;

			int label = (ignoreLabels ? 0 : lr[x]);
			label = (label >= sets.size()) ? 0 : label;
			const multi_img::Pixel& pixel = multi(y, x);
			BinSet &s = sets[label];

			BinSet::HashKey hashkey(boost::extents[multi.size()]);
			for (int d = 0; d < multi.size(); ++d) {
				int pos = floor(Compute::curpos(
									pixel[d], d, minval, binsize, illuminant));
				pos = std::max(pos, 0); pos = std::min(pos, nbins-1);
				hashkey[d] = (unsigned char)pos;
			}

			if (subtract) {
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
