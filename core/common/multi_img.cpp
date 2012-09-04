/*
	Copyright(c) 2012 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifdef WITH_OPENCV2 // theoretically, vole could be built w/o opencv..

#include "multi_img.h"
#include <iostream>
#include <string>
#include <vector>

using namespace std;

const float multi_img::ValueMin = -FLT_MAX;
const float multi_img::ValueMax = FLT_MAX;

multi_img::multi_img(const string& filename)
 : minval(0.), maxval(0.), width(0), height(0)
{
	read_image(filename);
	roi = cv::Rect(0, 0, width, height);
}

multi_img::multi_img(const cv::Mat& image) : minval(0.), maxval(0.)
{
	assert(!image.empty());
	/* we need to clone because opencv totally ignores const */
	read_mat(image.clone());
	roi = cv::Rect(0, 0, width, height);
	// init cache metadata
	resetPixels();
}

multi_img::multi_img(const cv::Mat& image, Value srcmin, Value srcmax)
 : minval(0.), maxval(0.)
{
	assert(!image.empty());
	read_mat(image.clone(), srcmin, srcmax);
	roi = cv::Rect(0, 0, width, height);
	// init cache metadata
	resetPixels();
}

multi_img & multi_img::operator=(const multi_img &a) {
	std::cerr << "multi_img: assignment" << std::endl;
	if (this != &a) {
		width = a.width; height = a.height;
		minval = a.minval; maxval = a.maxval;
		meta = a.meta;
		roi = a.roi;
		bands.resize(a.bands.size());
		for (size_t i = 0; i < bands.size(); ++i)
			bands[i] = a.bands[i].clone();
		resetPixels(true);
	}
	return *this;
}

void multi_img::init(int h, int w, size_t d, Value minv, Value maxv) {
	minval = minv; maxval = maxv;
	height = h; width = w;
	roi = cv::Rect(0, 0, 0, 0);
	meta.resize(d);
	bands.resize(d);
	for (size_t i = 0; i < d; i++)
		bands[i] = Band(h, w); // each band has distinct data array
	resetPixels();
}

multi_img::multi_img(int height, int width, size_t size)
{
	init(height, width, size);
}

multi_img::multi_img(const multi_img &a)
 : minval(a.minval), maxval(a.maxval), width(a.width), height(a.height),
   meta(a.meta), roi(a.roi), bands(a.size())
{
	std::cerr << "multi_img: copy" << std::endl;
	for (size_t i = 0; i < bands.size(); ++i)
		bands[i] = a.bands[i].clone();
	resetPixels();
}

multi_img::multi_img(const multi_img &a, const cv::Rect &roi)
 : minval(a.minval), maxval(a.maxval), width(roi.width), height(roi.height),
   meta(a.meta), roi(roi), bands(a.size())
{
	std::cerr << "multi_img: reference w/ roi" << std::endl;
	for (size_t i = 0; i < bands.size(); ++i)
		bands[i] = Band(a.bands[i], roi);
	/* FIXME: - inconsistent to other copy constr.
	          - will lead to corrupt cache data!
                use vector of pointers for cache and copy them, too? */
	resetPixels();
}

multi_img::multi_img(const multi_img &a, unsigned int start, unsigned int end)
 : minval(a.minval), maxval(a.maxval), width(a.width), height(a.height), roi(a.roi)
{
	assert(start < a.size() && end < a.size());
	std::cerr << "multi_img: reference w/ spectral crop" << std::endl;
	meta.insert(meta.begin(), a.meta.begin() + start, a.meta.begin() + (end+1));
    bands.insert(bands.begin(), a.bands.begin() + start, a.bands.begin() + (end+1));
	/* FIXME: - inconsistent to other copy constr.
	          - will lead to corrupt cache data!
			  read-only flag? no constructor but const method?
	*/
	resetPixels();
}

multi_img multi_img::clone(bool cloneCache) const
{
	multi_img ret(*this);
	if (cloneCache) {
		ret.pixels = pixels;
		ret.dirty = dirty.clone();
		ret.anydirt = anydirt;
	}
	return ret;
}

void multi_img::resetPixels(bool force) const
{
	if (force || pixels.empty()) {
		pixels.assign(width * height, Pixel(size()));
	}
	if (force || dirty.empty())
		dirty = Mask(height, width, 255);
	else
		dirty.setTo(255);
	anydirt = true;
}

void multi_img::rebuildPixels(bool optimistic) const
{
	if (!anydirt || (optimistic && countNonZero(dirty) == 0))
		return;

	cerr << "multi_img: complete rebuild" << endl;
	BandConstIt it;
	register unsigned int d, i;
	for (d = 0; d < size(); ++d) {
		const Band &src = bands[d];
		for (it = src.begin(), i = 0; it != src.end(); ++it, ++i)
			pixels[i][d] = *it;
	}
	dirty.setTo(0);
	anydirt = false;
}

void multi_img::rebuildPixel(unsigned int row, unsigned int col) const
{
	cerr << "multi_img: rebuild pixel " << row << "." << col << endl;
	Pixel &p = pixels[row*width + col];
	for (size_t i = 0; i < size(); ++i)
		p[i] = bands[i](row, col);

	dirty(row, col) = 0;
}

std::vector<const multi_img::Pixel*> multi_img::getSegment(const Mask &mask)
{
	assert(mask.rows == height && mask.cols == width);

	std::vector<const Pixel*> ret;
	for (int row = 0; row < height; ++row) {
		const uchar *m = mask[row];
		for (int col = 0; col < width; ++col) {
			if (m[col] > 0) {
				if (anydirt && dirty(row, col))
					rebuildPixel(row, col);
				ret.push_back(&pixels[row*width + col]);
			}
		}
	}
	return ret;
}

std::vector<multi_img::Pixel> multi_img::getSegmentCopy(const Mask &mask)
{
	assert(mask.rows == height && mask.cols == width);

	std::vector<Pixel> ret;
	for (int row = 0; row < height; ++row) {
		const uchar *m = mask[row];
		for (int col = 0; col < width; ++col) {
			if (m[col] > 0) {
				if (anydirt && dirty(row, col))
					rebuildPixel(row, col);
				ret.push_back(pixels[row*width + col]);
			}
		}
	}
	return ret;
}

void multi_img::setPixel(unsigned int row, unsigned int col,
						 const Pixel &values)
{
	assert((int)row < height && (int)col < width);
	assert(values.size() == size());
	Pixel &p = pixels[row*width + col];
	p = values;
	for (size_t i = 0; i < size(); ++i)
		bands[i](row, col) = p[i];

	dirty(row, col) = 0;
}

// matrix version
void multi_img::setPixel(unsigned int row, unsigned int col,
						 const cv::Mat_<Value>& values)
{
	assert((int)row < height && (int)col < width);
	assert(values.rows*values.cols == (int)size());
	Pixel &p = pixels[row*width + col];
	p.assign(values.begin(), values.end());

	for (size_t i = 0; i < size(); ++i)
		bands[i](row, col) = p[i];

	dirty(row, col) = 0;
}

void multi_img::setBand(unsigned int band, const Band &data,
						const Mask &mask)
{
	assert(band < size());
	assert(data.rows == height && data.cols == width);
	Band &b = bands[band];
	BandConstIt bit = b.begin();
	MaskConstIt dit = dirty.begin();
	/* we use opencv to copy the band data. afterwards, we update the pixels
	   cache. we do this only for pixels, which are not dirty yet (and would
	   need a complete rebuild anyways. As we instantly fix the other pixels,
	   those do not get marked as dirty by us. */
	if (!mask.empty()) {
		assert(mask.rows == height && mask.cols == width);
		MaskConstIt mit = mask.begin();
		data.copyTo(b, mask);
		for (int i = 0; bit != b.end(); ++bit, ++dit, ++mit, ++i)
			if ((*mit > 0)&&(*dit == 0))
				pixels[i][band] = *bit;
	} else {
		data.copyTo(b);
		for (int i = 0; bit != b.end(); ++bit, ++dit, ++i) {
			if ((*dit == 0))
				pixels[i][band] = *bit;
		}
	}
}

void multi_img::setSegment(const std::vector<Pixel> &values, const Mask &mask)
{
	assert(mask.rows == height && mask.cols == width);
	int i = 0;
	for (int row = 0; row < height; ++row) {
		const uchar *m = mask[row];
		for (int col = 0; col < width; ++col) {
			if (m[col] > 0) {
				setPixel(row, col, values.at(i)); // vector does bound checking
				++i;
			}
		}
	}
}

void multi_img::setSegment(const std::vector<cv::Mat_<Value> > &values,
						   const Mask &mask)
{
	assert(mask.rows == height && mask.cols == width);
	int i = 0;
	for (int row = 0; row < height; ++row) {
		const uchar *m = mask[row];
		for (int col = 0; col < width; ++col) {
			if (m[col] > 0) {
				setPixel(row, col, values[i]);
				++i;
			}
		}
	}
}

void multi_img::setTo(const Pixel &p)
{
	assert(p.size() == size());
	for (size_t i = 0; i < size(); ++i)
		bands[i].setTo(p[i]);
}

void multi_img::applyCache()
{
	for (unsigned int d = 0; d < size(); ++d) {
		Band &dst = bands[d];
		BandIt it;
		unsigned int i;
		for (it = dst.begin(), i = 0; it != dst.end(); it++, i++)
			*it = pixels[i][d];
	}
	// cache data is now consistent with band data
	dirty.setTo(0);
	anydirt = false;
}

std::pair<multi_img::Value, multi_img::Value>
multi_img::data_range(double fraction) const
{
	assert(!empty());
	assert(fraction < .5);

	/*  find overall data range */
	std::pair<Value, Value> ret = make_pair(bands[0](0,0), bands[0](0,0));
	double tmp1, tmp2;
	for (unsigned int d = 0; d < size(); ++d) {
		cv::minMaxLoc(bands[d], &tmp1, &tmp2);
		ret.first = std::min<Value>(ret.first, (Value)tmp1);
		ret.second = std::max<Value>(ret.second, (Value)tmp2);
	}

	if (fraction == 0.) {
		return ret;
	}

	/* we build a histogram to find "good" data range */
	cv::Mat_<float> hist;
	// todo: argument of the function?
	int bins = 100;
	int histSize[] = { bins };
	float range1[] = { ret.first, ret.second };
	const float *ranges[] = { range1 };
	int channels[] = { 0 };

	for (unsigned int d = 0; d < size(); ++d)
		cv::calcHist(&bands[d], 1, channels, cv::Mat(), hist,
					 1, histSize, ranges, true, d>0);

	/* we defensively choose bin borders as new range approx. */
	double binsize = (ret.second - ret.first)/(double)bins;
	int needed = (int)std::ceil((double)(width*height*size())*fraction);
	int found, index;

	/* first: small values */
	found = 0;
	index = 0;
	while (found < needed) {
		found += (int)hist[index][0];
		index++;
	}
	// set to lower boundary of last outlier bin
	ret.first = ret.first + (Value)(binsize*(index - 1));

	/* second: large values */
	found = 0;
	index = hist.rows - 1;
	while (found < needed) {
		found += (int)hist[index][0];
		index--;
	}
	// set to upper boundary of last outlier bin
	ret.second = ret.second - (Value)(binsize*(hist.rows - index - 2));

	return ret;
}

cv::PCA multi_img::pca(unsigned int components) const
{
	assert(components <= size());

	// make sure cache is there
	rebuildPixels(true);

	// create input matrix
	cv::Mat_<Value> input(size(), pixels.size());
	for (size_t i = 0; i < pixels.size(); ++i) {
		cv::Mat_<Value> in(pixels[i]);
		cv::Mat_<Value> out(input.col(i));
		in.copyTo(out);
	}

	// perform PCA
	cv::PCA ret(input, cv::noArray(), CV_PCA_DATA_AS_COL, components);
	return ret;
}

multi_img multi_img::project(const cv::PCA &pca) const
{
	multi_img ret(height, width, pca.eigenvectors.rows);

	// make sure input cache is there
	rebuildPixels(true);

	// write
	for (size_t i = 0; i < ret.pixels.size(); ++i) {
		cv::Mat_<Value> input(pixels[i]);
		cv::Mat_<Value> output(ret.pixels[i]);
		pca.project(input, output);
	}

	// write back to band data
	ret.applyCache();

	// set min/max as observed
	std::pair<Value, Value> range = ret.data_range();
	ret.minval = range.first;
	ret.maxval = range.second;

	return ret;
}

multi_img::Pixel multi_img::project(const Pixel &p, const cv::PCA &pca)
{
	assert(p.size() == pca.eigenvectors.cols);
	Pixel ret(pca.eigenvectors.rows);
	cv::Mat_<Value> input(p);
	cv::Mat_<Value> output(ret);

	pca.project(input, output);
	return ret;
}

multi_img::Pixel multi_img::backProject(const Pixel &p, const cv::PCA &pca)
{
	assert(p.size() == pca.eigenvectors.rows);
	Pixel ret(pca.eigenvectors.cols);
	cv::Mat_<Value> input(p);
	cv::Mat_<Value> output(ret);

	pca.backProject(input, output);
	return ret;
}

void multi_img::clamp()
{
	for (unsigned int d = 0; d < size(); ++d) {
		Band &b = bands[d];
		cv::max(b, minval, b);
		cv::min(b, maxval, b);
	}
	// cache became invalid
	resetPixels();
}

void multi_img::data_rescale(Value newminval, Value newmaxval)
{
	if (minval == newminval && maxval == newmaxval)
		return;

	Value scale = (newmaxval - newminval)/(maxval - minval);
	for (size_t d = 0; d < size(); ++d) {
		Band &b = bands[d];
		if (newminval == 0. && minval == 0.) {
			b *= scale;
		} else {
			b = (b - minval) * scale;
			if (newminval != 0.)
				b += newminval;
		}
	}
	minval = newminval; maxval = newmaxval;
	// cache became invalid
	resetPixels();
}

void multi_img::data_stretch()
{
	// remember theoretical data range
	Value newmin = minval, newmax = maxval;

	// find actual data range
	std::pair<Value, Value> current = data_range();

	// correct the current range information (used by data_rescale())
	minval = current.first; maxval = current.second;

	// perform rescaling
	data_rescale(newmin, newmax);
}

void multi_img::data_stretch_single(Value newmin, Value newmax)
{
	if (newmin != newmax) {
		minval = newmin;
		maxval = newmax;
	}
	for (size_t d = 0; d < size(); ++d) {
		Band &b = bands[d];
		double mi, ma;
		cv::minMaxLoc(b, &mi, &ma);
/*		double scale = (maxval - minval)/(ma - mi), shift = -mi;

		if (newminval == 0. && minval == 0.) {
			b *= scale;
		} else {
			b = (b - minval) * scale;
			if (newminval != 0.)
				b += newminval;
		}*/
	}
	// cache became invalid
	resetPixels();
}

void multi_img::flip(int flipCode)
{
	for (size_t i = 0; i < size(); ++i)
		cv::flip(bands[i], bands[i], flipCode);

	// cache became invalid
	resetPixels();
}

void multi_img::transpose()
{
	for (size_t i = 0; i < size(); ++i) {
		Band tmp;
		cv::transpose(bands[i], tmp);
		bands[i] = tmp;
	}

	// geometry change
	std::swap(width, height);

	// cache became invalid, and geometry change
	resetPixels(true);
}

void multi_img::apply_logarithm()
{
	for (size_t i = 0; i < size(); ++i) {
		// will assign large negative value to 0 pixels
		cv::log(bands[i], bands[i]);
		// get rid of negative values (when pixel value was 0)
		bands[i] = cv::max(bands[i], 0.);
		//cv::max(bands[i], 0., bands[i]); // bug in OpenCV 2.3.1
	}
	// data format has changed as follows
	minval = 0.;
	maxval = log(maxval);
	// cache became invalid
	resetPixels();
}

void multi_img::blur(cv::Size ksize, double sigmaX, double sigmaY,
					 int borderType)
{
	for (size_t i = 0; i < size(); ++i) {
		cv::GaussianBlur(bands[i], bands[i], ksize, sigmaX, sigmaY, borderType);
	}
	// cache became invalid
	resetPixels();
}

#endif // opencv
