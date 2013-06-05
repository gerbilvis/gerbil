#include "viewer_tasks.h"

#include <multi_img.h>

#include <tbb/blocked_range2d.h>
#include <tbb/task.h>

///////////////////////////////
/// fillMaskSingleBody
///////////////////////////////

fillMaskSingleBody::fillMaskSingleBody(
		cv::Mat1b &mask, const
		multi_img::Band &band, int dim, int sel,
		multi_img::Value minval,
		multi_img::Value binsize,
		const std::vector<multi_img::Value>
		&illuminant)
	: mask(mask), band(band), dim(dim), sel(sel), minval(minval),
	  binsize(binsize), illuminant(illuminant)
{
}

void fillMaskSingleBody::operator ()(
		const tbb::blocked_range2d<size_t> &r) const
{
	for (int y = r.rows().begin(); y != r.rows().end(); ++y) {
		unsigned char *mrow = mask[y];
		const multi_img::Value *brow = band[y];
		for (int x = r.cols().begin(); x != r.cols().end(); ++x) {
			int pos = floor(
					Compute::curpos(brow[x], dim, minval, binsize, illuminant));
			mrow[x] = (pos == sel) ? 1 : 0;
		}
	}
}

///////////////////////////////
/// fillMaskLimitersBody
///////////////////////////////

fillMaskLimitersBody::fillMaskLimitersBody(
		cv::Mat1b &mask, const multi_img &image, multi_img::Value
		minval, multi_img::Value binsize, const std::vector<multi_img::Value>
		&illuminant, const std::vector<std::pair<int, int> > &l)
	: mask(mask), image(image), minval(minval), binsize(binsize),
	illuminant(illuminant), l(l)
{
}

void fillMaskLimitersBody::operator ()(
		const tbb::blocked_range2d<size_t> &r) const
{
	for (int y = r.rows().begin(); y != r.rows().end(); ++y) {
		unsigned char *row = mask[y];
		for (int x = r.cols().begin(); x != r.cols().end(); ++x) {
			row[x] = 1;
			const multi_img::Pixel &p = image(y, x);
			for (unsigned int d = 0; d < image.size(); ++d) {
				int pos = floor(Compute::curpos(
									p[d], d, minval, binsize, illuminant));
				if (pos < l[d].first || pos > l[d].second) {
					row[x] = 0;
					break;
				}
			}
		}
	}
}

///////////////////////////////
/// updateMaskLimitersBody
///////////////////////////////

updateMaskLimitersBody::updateMaskLimitersBody(
		cv::Mat1b &mask, const multi_img &image, int dim,
		multi_img::Value minval, multi_img::Value binsize, 
		const std::vector<multi_img::Value> &illuminant, 
		const std::vector<std::pair<int, int> > &l)
	: mask(mask), image(image), dim(dim), minval(minval),
	binsize(binsize), illuminant(illuminant), l(l)
{
}

void updateMaskLimitersBody::operator ()(
		const tbb::blocked_range2d<size_t> &r) const
{
	for (int y = r.rows().begin(); y != r.rows().end(); ++y) {
		unsigned char *mrow = mask[y];
		const multi_img::Value *brow = image[dim][y];
		for (int x = r.cols().begin(); x != r.cols().end(); ++x) {
			int pos = floor(Compute::curpos(
								brow[x], dim, minval, binsize, illuminant));
			if (pos < l[dim].first || pos > l[dim].second) {
				mrow[x] = 0;
			} else if (mrow[x] == 0) { // we need to do exhaustive test
				mrow[x] = 1;
				const multi_img::Pixel& p = image(y, x);
				for (unsigned int d = 0; d < image.size(); ++d) {
					int pos = floor(Compute::curpos(
										p[d], d, minval, binsize, illuminant));
					if (pos < l[d].first || pos > l[d].second) {
						mrow[x] = 0;
						break;
					}
				}
			}
		}
	}
}
