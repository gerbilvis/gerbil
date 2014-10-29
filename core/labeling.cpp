/*	
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/
#ifdef WITH_OPENCV2 // theoretically, vole could be built w/o opencv..

#include "labeling.h"
#include "qtopencv.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>

Labeling::Labeling(const cv::Mat &labeling, bool binary)
	: yellowcursor(true), shuffle(false), shuffleV(false)
{
	read(labeling, binary);
}

Labeling::Labeling(const std::string &filename, bool binary)
	: yellowcursor(true), shuffle(false), shuffleV(false)
{
	cv::Mat src = cv::imread(filename, cv::IMREAD_UNCHANGED);
	if (src.empty()) {
		std::cerr << "ERROR: Failed to load " << filename << std::endl;
		return;
	}
	read(src, binary);
}

void Labeling::setLabels(const cv::Mat &src)
{
	labels = src;
	double tmp1, tmp2;
	cv::minMaxLoc(labels, &tmp1, &tmp2);
	labelcount = tmp2 + 1; // don't forget 0.
	if (labelColors.size() != (size_t)labelcount)
		labelColors.clear();
}

void Labeling::read(const cv::Mat &src, bool binary)
{
	if (binary) {
		readBinary(src);
		return;
	}

	if (src.channels() > 1) { // RGB(A) input
		assert((src.channels() == 3 || src.channels() == 4)
			   && src.depth() == CV_8U);
		cv::Mat3b src3b;
		if (src.channels() == 4) { // RGBA case (no sense, but they exist!)
			// only take first three channels
			src3b.create(src.rows, src.cols); // allocation mandatory!
			int pairing[] = { 0,0, 1,1, 2,2 };
			cv::mixChannels(&src, 1, &src3b, 1, pairing, 3);
		} else {
			src3b = src;
		}
		read(src3b);
		return;
	}

	/** one-channel case **/

	// determine highest possible intensity
	int bins = 256;
	if (src.depth() != CV_8U) {
		double tmp1, tmp2;
		cv::minMaxLoc(src, &tmp1, &tmp2);
		bins = tmp2 + 1;
	}
	// convert to a strict format for convenience -- we don't expect >16 bit
	cv::Mat1w src1w = src;
	read(src1w, bins);
}

struct Vec3bCompare {
	bool operator() (const cv::Vec3b &a, const cv::Vec3b &b) const
	{
		if (a[0] == b[0]) {
			if (a[1] == b[1]) {
				return (a[2] < b[2]);
			}
			return a[1] < b[1];
		}
		return a[0] < b[0];
	}
};

void Labeling::read(const cv::Mat3b &src)
{
	std::map<cv::Vec3b, short, Vec3bCompare> palette;
	palette[cv::Vec3b(0, 0, 0)] = -1; // always have black background label

	// find all colors
	cv::Mat3b::const_iterator its;
	for (its = src.begin(); its != src.end(); ++its)
		palette[*its] = -1;

	// assign labelColors based on map order (somewhat canonical label indices)
	labelColors.clear();
    std::map<cv::Vec3b, short, Vec3bCompare>::iterator itp;
	for (itp = palette.begin(); itp != palette.end(); ++itp)
	{
		itp->second = labelColors.size(); // index into labelColors
		labelColors.push_back(itp->first);
	}
	labelcount = labelColors.size();

	// assign the color indices to the label matrix
	labels = cv::Mat1s(src.rows, src.cols);
	cv::Mat1s::iterator itl;
	for (its = src.begin(), itl = labels.begin();
		 its != src.end(); ++its, ++itl) {
		*itl = palette[*its];
	}

	/* Special case: only one white label stored in RGB image (stupid).
	   We don't want to use white color, it confuses the user. */
	if ((labelcount == 2) && (labelColors[1][0] == labelColors[1][1])
						  && (labelColors[1][1] == labelColors[1][2]))
		labelColors[1] = cv::Vec3b(0, 255, 0);
}

void Labeling::read(const cv::Mat1w &src, int bins)
{
	/* find all used intensities */
	std::vector<int> indices(bins, 0);
	// always have a background label
	indices[0] = -1;
	cv::Mat1w::const_iterator its;
	for (its = src.begin(); its != src.end(); ++its)
		indices[*its] = -1;

	/* assign indices */
	labelcount = 0;
	for (int i = 0; i < bins; ++i) {
		if (indices[i] == -1)
			indices[i] = labelcount++;
	}

	/* if needed, clear labelColors */
	if (labelColors.size() < (size_t)labelcount)
		labelColors.clear();

	/* assign the intensity indices to the label matrix */
	labels = cv::Mat1s(src.rows, src.cols);
	cv::Mat1s::iterator itl;
	for (its = src.begin(), itl = labels.begin();
		 its != src.end(); ++its, ++itl) {
		*itl = indices[*its];
	}
}

void Labeling::readBinary(const cv::Mat &src)
{
	if (src.channels() > 1) {
		std::vector<cv::Mat> array;
		cv::split(src, array);
		for (size_t i = 1; i < array.size(); ++i)
			cv::max(array[i], array[0], array[0]);
		labels = array[0]; // implicit conversion to Mat1s
	} else {
		labels = src; // implicit conversion to Mat1s
	}

	// treat all values != 0 as 1
	labels = labels > 0;

	labelColors.clear();
	labelColors.push_back(cv::Vec3b(  0,   0,   0));
	labelColors.push_back(cv::Vec3b(  0, 255,   0));
	labelcount = 2;
}

void Labeling::setColors(const std::vector<cv::Vec3b> &colors)
{
	assert(colors.size() >= (size_t)labelcount);
	labelColors = colors;
}

const std::vector<cv::Vec3b>& Labeling::colors() const
{
	if (labelColors.empty())
		buildColors();

	return labelColors;
}

cv::Mat1b Labeling::grayscale() const
{
	assert(labelcount <= 256);
	cv::Mat1b ret = labels*(255./(double)(labelcount - 1));
	return ret;
}

cv::Mat3b Labeling::bgr() const
{
	if (labelColors.empty())
		buildColors();

	cv::Mat3b ret(labels.rows, labels.cols);
	for (int y = 0; y < labels.rows; ++y) {
		cv::Vec3b *rety = ret[y];
		const short *lbly = labels[y];
		for (int x = 0; x < labels.cols; ++x) {
			rety[x] = labelColors[lbly[x]];
		}
	}
	return ret;
}

void Labeling::consolidate()
{
	int idx = 1, // consolidation starts at 1
		maxlabel = 0; // if all labels are empty
	std::vector<int> index(labelcount);
	for (int i = 1; i < labelcount; ++i) {
		if (cv::sum(labels == i)[0] == 0) {
			// label is empty
			index[i] = -1;
		} else {
			index[i] = idx++;
			maxlabel = index[i];
		}
	}
	for (int i = 1; i < labelcount; ++i) {
		if (index[i] != -1) {
			labels.setTo(index[i], (labels == i));
		}
	}
	labelcount = maxlabel + 1;
	buildColors();
}

void Labeling::buildColors() const
{
	labelColors = colors(labelcount, yellowcursor, shuffleV);
	if (shuffle) {
		// shuffle everything but first color (black)
		std::random_shuffle(labelColors.begin() + 1, labelColors.end());
	}
}

std::vector<cv::Vec3b> Labeling::colors(int count, bool yellowcursor,
										bool shuffleV)
{
	std::vector<cv::Vec3b> ret;
	if (count <= (yellowcursor ? 6 : 7)) {
		// standard set of label colors
		const cv::Vec3b standard[] =
		{	cv::Vec3b(  0,   0,   0), // 0 is index for unlabeled
			cv::Vec3b(  0, 255,   0), cv::Vec3b(  0,   0, 255),
			cv::Vec3b(255, 255,   0), cv::Vec3b(255,   0, 255),
			cv::Vec3b(  0, 255, 255), // 5: yellow
			cv::Vec3b(255,   0,   0) }; // blue is last, worst visible
		int c = 0;
		do {
			if (c == 5 && yellowcursor)
				++c;
			ret.push_back(standard[c]);
		} while (++c < count);
	} else {
		/* work out colors in float type, as opencv is not comfortable with
		   hsv input in uchar. it still assumes hue range [0, 360], lol! */
		cv::Mat3f hsvmap(count, 1);
		hsvmap[0][0] = cv::Vec3f(0.f, 0.f, 0.f); // 0 is index for unlabeled
		float distance, hue;
		if (yellowcursor) {
			// start at yellow (60°) + 1 step, stop at yellow - 1 step
			distance = 360.f / (float)(count);
			hue = 60.f + distance;
		} else {
			// start at red, stop at red - 1 step
			distance = 360.f / (float)(count - 1);
			hue = 0.f;
		}
		float val = 1.f;
		cv::RNG rng(cv::theRNG());
		for (int i = 1; i < count; i++) {
			if (shuffleV)
				val = rng.uniform(0.5f, 1.0f);
			hsvmap[0][i] = cv::Vec3f(hue, 1.f, val);
			hue += distance;
		}
		cv::cvtColor(hsvmap, hsvmap, CV_HSV2BGR);

		// save in ret vector (build matrix around vector to avoid a copy)
		ret.resize(count);
		cv::Mat3b dest(ret);
		dest = hsvmap*255.f;
	}
	return ret;
}

#endif // opencv
