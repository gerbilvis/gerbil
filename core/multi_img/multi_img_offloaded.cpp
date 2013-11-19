#include "multi_img_offloaded.h"
#include <opencv2/highgui/highgui.hpp>

using namespace std;

multi_img_offloaded::multi_img_offloaded(const vector<string> &files, const vector<BandDesc> &descs)
{
	int channels = 0;
	width = 0;
	height = 0;

	for (size_t fi = 0; fi < files.size(); ++fi) {
		cv::Mat src = cv::imread(files[fi], -1); // flag -1: preserve format

		if (src.empty()) {
			cerr << "ERROR: Failed to load " << files[fi] << endl;
			continue;
		}

		// test spatial size
		if (width > 0 && (src.cols != width || src.rows != height)) {
			cerr << "ERROR: Size mismatch for image " << files[fi] << endl;
			continue;
		}

		// set spatial size
		width = src.cols;
		height = src.rows;

		/* default to our favorite range */
		minval = MULTI_IMG_MIN_DEFAULT;
		maxval = MULTI_IMG_MAX_DEFAULT;

		// split & add everything in at the end
		channels = src.channels();
		if (channels > 1) {
			std::vector<cv::Mat> splitted(channels);
			cv::split(src, splitted);
			for (int c = 0; c < splitted.size(); ++c)
				bands.push_back(make_pair(files[fi], c));
		} else {
			bands.push_back(make_pair(files[fi], 0));
		}

		cout << "Added " << files[fi] << ":\t" << channels
			 << (channels == 1 ? " channel, " : " channels, ")
			 << (src.depth() == CV_16U ? 16 : 8) << " bits";
		if (descs.empty() || descs[fi].empty)
			cout << endl;
		else
			cout << ", " << descs[fi].center << " nm" << endl;
	}

	/* add meta information if present. */
	if (!descs.empty()) {
		assert(meta.size() + descs.size() == bands.size());
		meta.insert(meta.end(), descs.begin(), descs.end());
	} else {
		/* Hack: when input was single RGB image, we assume RGB peak wavelengths
				 (from Hamamatsu) to enable re-calculation of RGB image */
		// NOTE: for this to work as expected, incoming data still needs to
		//	have linear response, which is not true for typical RGB imaging
		if (files.size() == 1 && channels == 3) {
			meta.push_back(BandDesc(460));
			meta.push_back(BandDesc(540));
			meta.push_back(BandDesc(620));
		} else {
			meta.resize(bands.size());
		}
	}

	if (bands.size())
		cout << "Total of " << bands.size() << " bands. "
			 << "Spatial size: " << width << "x" << height
			 << "   (" << bands.size()*width*height*sizeof(Value)/1048576. << " MB)" << endl;
}

unsigned int multi_img_offloaded::size() const
{
	return (unsigned int)bands.size();
}

bool multi_img_offloaded::empty() const
{
	return bands.empty();
}

void multi_img_offloaded::scopeBand(const Band &source, const cv::Rect &roi, Band &target) const
{
	Band scoped(source, roi);
	target = scoped.clone();
}

void multi_img_offloaded::getBand(size_t band, Band &data) const
{
	cv::Mat src = cv::imread(bands[band].first, -1); // flag -1: preserve format

	if (src.empty()) {
		cerr << "ERROR: Failed to load " << bands[band].first << endl;
		return;
	}

	// find original data range, we assume minimum is 0
	Value srcminval = 0.;
	Value srcmaxval;
	// we expect CV_8U, CV_16U or floating point in [0..1]
	switch (src.depth()) {
		case CV_8U:	 { srcmaxval = 255.; break; }
		case CV_16U: { srcmaxval = 65535.; break; }
		case CV_32F:
		case CV_64F: { srcmaxval = 1.; break; }
		default:	assert(42 == 0);	// we don't handle other formats so far!
	}

	// convert to right datatype, scaling
	cv::Mat tmp;
	src.convertTo(tmp, ValueType);

	// rescale data accordingly
	if (srcminval == 0. && minval == 0.) {
		if (maxval != srcmaxval)
			tmp *= maxval/srcmaxval;
	} else {
		Value scale = (maxval - minval)/(srcmaxval - srcminval);
		tmp = (tmp - srcminval) * scale;
		if (minval != 0.)
			tmp += minval;
	}

	// split & add everything in at the end
	size_t cc = tmp.channels();
	if (cc > 1) {
		std::vector<Band> channels(cc);
		cv::split(tmp, channels);
		data = channels[bands[band].second];
	} else {
		data = tmp;
	}
}
