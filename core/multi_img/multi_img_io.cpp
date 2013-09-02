/*	
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifdef WITH_OPENCV2 // theoretically, vole could be built w/o opencv..

#include "multi_img.h"
#include "qtopencv.h"

#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

unsigned short* multi_img::export_interleaved(bool useDataRange) const
{
	rebuildPixels();

	Range range(minval, maxval);
	if (useDataRange) {
		// determine actual minval/maxval
		range = data_range();
	}

	Value scale = 65535.0/(range.max - range.min);
	int length = size(), i = 0, d;
	unsigned short *ret = new unsigned short[length*width*height];

	for (vector<Pixel>::const_iterator it = pixels.begin(); it != pixels.end(); ++it)
		for (d = 0; d < length; ++d)
			ret[i++] = ((*it)[d] - range.min) * scale;

	return ret;
}

#ifdef WITH_QT
// exports one band
QImage multi_img::export_qt(unsigned int band) const
{
	assert(band < size());
	return vole::Band2QImage(bands[band], minval, maxval);
}
#endif

// read image part (what fits into one cv::Mat)
int multi_img::read_mat(const cv::Mat &src)
{
	// find original data range, we assume minimum is 0
	Value srcmaxval;
	// we expect CV_8U, CV_16U or floating point in [0..1]
	switch (src.depth()) {
		case CV_8U:	 { srcmaxval = 255.; break; }
		case CV_16U: { srcmaxval = 65535.; break; }
		case CV_32F:
		case CV_64F: { srcmaxval = 1.; break; }
		default:	assert(42 == 0);	// we don't handle other formats so far!
	}
	return read_mat(src, 0., srcmaxval);
}

// read image part (what fits into one cv::Mat)
int multi_img::read_mat(const cv::Mat &src, Value srcminval, Value srcmaxval)
{
	if (minval == maxval) { // i.e. uninitialized
		/* default to our favorite range */
		minval = MULTI_IMG_MIN_DEFAULT; maxval = MULTI_IMG_MAX_DEFAULT;
	}

	// set spatial size
	width = src.cols;
	height = src.rows;

	// convert to right datatype, scaling
	cv::Mat tmp;
	src.convertTo(tmp, ValueType);

	// rescale data accordingly
	if (srcminval == 0. && minval == 0.) {
		if (maxval != srcmaxval) // evil comparison is o.k. here
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
		bands.insert(bands.end(), channels.begin(), channels.end());
	} else {
		bands.push_back(tmp);
	}
	return cc;
}

// read multires. image into vector
void multi_img::read_image(const vector<string> &files, const vector<BandDesc> &descs)
{
	int channels = 0;
	
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

		channels = read_mat(src);
		
	    cout << "Added " << files[fi] << ":\t" << channels
             << (channels == 1 ? " channel, " : " channels, ")
			 << (src.depth() == CV_16U ? 16 : 8) << " bits";
		if (descs.empty() || descs[fi].empty)
			cout << endl;
		else
			cout << ", " << descs[fi].center << " nm" << endl;
	}

	/* invalidate pixel cache as pixel length has changed
	   This step is _mandatory_ also to initialize cache containers */
	pixels.clear();
	resetPixels();

	/* add meta information if present. */
	if (!descs.empty()) {
		assert(meta.size() + descs.size() == size());
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
			meta.resize(size());
		}
	}

	if (size())
		cout << "Total of " << size() << " bands. "
			 << "Spatial size: " << width << "x" << height
			 << "   (" << size()*width*height*sizeof(Value)/1048576. << " MB)" << endl;
}

#endif // opencv
