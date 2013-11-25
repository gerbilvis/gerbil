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

using namespace std;

namespace vole {

	Labeling::Labeling(const cv::Mat &labeling, bool binary)
		: yellowcursor(true), shuffle(false), shuffleV(false)
{
	read(labeling, binary);
}

Labeling::Labeling(const string &filename, bool binary)
	: yellowcursor(true), shuffle(false), shuffleV(false)
{
	cv::Mat src = cv::imread(filename, -1); // flag -1: preserve format
	if (src.empty()) {
		cerr << "ERROR: Failed to load " << filename << endl;
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
		// treat all values != 0 as 1
		if (src.channels() > 1) {
			vector<cv::Mat> array;
			cv::split(src, array);
			for (size_t i = 1; i < array.size(); ++i)
				cv::max(array[i], array[0], array[0]);
			labels = array[0];
		} else {
			labels = src;
		}
		labels = labels > 0;
		labelColors.clear();
		labelColors.push_back(cv::Vec3b(  0,   0,   0));
		labelColors.push_back(cv::Vec3b(255, 255, 255));
		labelcount = 2;
	} else {
		float maxv = (src.depth() == CV_16S ? 32767.f : 255.f);
		if (src.depth() != CV_8U) {
			double tmp1, tmp2;
			cv::minMaxLoc(src, &tmp1, &tmp2);
			maxv = tmp2;
		}
		int bins = maxv + 1;
		if (src.channels() == 1) {
			vector<int> indices(bins, 0);
			for (int y = 0; y < src.rows; ++y) {
				for (int x = 0; x < src.cols; ++x) {
					int intensity;
					if (src.depth() == CV_16S)
						intensity = src.at<short>(y, x);
					else if (src.depth() == CV_8U)
						intensity = src.at<uchar>(y, x);
					else
						intensity = src.at<int>(y, x);
					indices[intensity]++;
				}
			}
			int index = 0;
			for (int i = 0; i < bins; ++i) {
				if (indices[i] > 0) {
					indices[i] = index;
					index++;
				}
			}
			labelcount = index;
			if (labelColors.size() < (size_t)labelcount)
				labelColors.clear();
			labels = cv::Mat1s(src.rows, src.cols);
			for (int y = 0; y < src.rows; ++y) {
				for (int x = 0; x < src.cols; ++x) {
					int intensity;
					if (src.depth() == CV_16S)
						intensity = src.at<short>(y, x);
					else if (src.depth() == CV_8U)
						intensity = src.at<uchar>(y, x);
					else
						intensity = src.at<int>(y, x);
					labels(y, x) = indices[intensity];
				}
			}
		} else { // rgb images
			assert((src.channels() == 3 || src.channels() == 4)
				   && bins == 256);
			cv::Mat3b src3b;
			if (src.channels() == 4) { // RGBA case (no sense, but they exist!)
				// only take first three channels
				src3b.create(src.rows, src.cols); // allocation mandatory!
				int pairing[] = { 0,0, 1,1, 2,2 };
				cv::mixChannels(&src, 1, &src3b, 1, pairing, 3);
			} else {
				src3b = src;
			}

			// calculate histogram of the image colors, stored in indices[b,g,r]
			vector<vector<vector<int> > > indices
					(bins, vector<vector<int> >(bins, vector<int>(bins, 0)));
			cv::Mat3b::iterator its;
			for (its = src3b.begin(); its != src3b.end(); ++its) {
				const cv::Vec3b &v = *its;
				indices[v[0]][v[1]][v[2]]++;
			}

			// find colors in the rgb image and add them as label colors
			// at the same time, replace the frequency of the color by its index
			// FIXME: 256^3 iterations for adding 1 - 10 label colors?
			//		  use map or similar structure instead -> bugtracker #26
			labelColors.clear();
			int index = 0;
			for (int b = 0; b < bins; ++b) {
				for (int g = 0; g < bins; ++g) {
					for (int r = 0; r < bins; ++r) {
						if (indices[b][g][r] > 0) {
							indices[b][g][r] = index;
							labelColors.push_back(cv::Vec3b(b, g, r));
							index++;
						}
					}
				}
			}
			labelcount = index;

			// assign the color indices to the label matrix
			labels = cv::Mat1s(src3b.rows, src3b.cols);
			cv::Mat1s::iterator itl;
			for (its = src3b.begin(), itl = labels.begin();
				 its != src3b.end(); ++its, ++itl) {
				const cv::Vec3b &v = *its;
				*itl = indices[v[0]][v[1]][v[2]];
			}
			/* Special case: only one white label stored in RGB image (stupid).
			   We don't want to use white color, it confuses the user. */
			if ((labelcount == 2) && (labelColors[1][0] == labelColors[1][1])
			                      && (labelColors[1][1] == labelColors[1][2]))
				labelColors[1] = cv::Vec3b(0, 255, 0);
		}
	}
}

void Labeling::setColors(const vector<cv::Vec3b> &colors)
{
	assert(colors.size() >= (size_t)labelcount);
	labelColors = colors;
}

const vector<cv::Vec3b>& Labeling::colors() const
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

vector<cv::Vec3b> Labeling::colors(int count, bool yellowcursor, bool shuffleV)
{
	vector<cv::Vec3b> ret;
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

}
#endif // opencv
