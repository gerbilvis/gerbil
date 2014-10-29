#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>
#include "mfams.h"

using namespace std;

namespace seg_meanshift {

bool FAMS::loadPoints(char* filename) {
	bgLog("Load data points from P3 NetPBM file %s... ", filename);
	FILE * fd;
	char head[255];
	int  bpp;
	fd = fopen(filename, "r");

	if (!fd) {
		bgLog("Error opening %s\n", filename);
		return false;
	}
	if ((fscanf(fd, "%s %u %u %d", head, &w_, &h_, &bpp) == 4)
		&& (strcmp(head, "P3") == 0) && (bpp == 255)) {
		n_ = w_ * h_;
		d_ = 3;
	} else {
		bgLog("Error reading %s\n", filename);
		fclose(fd);
		return false;
	}

	// read data into temp vector
	std::vector<std::vector<float> > temp(n_, std::vector<float>(d_));
	for (size_t i = 0; i < temp.size(); ++i) {
		for (size_t j = 0; j < temp[i].size(); ++j) {
			if (fscanf(fd, "%f", &temp[i][j]) != 1) {
				bgLog("Error reading %s\n", filename);
				fclose(fd);
				return false;
			}
		}
	}
	fclose(fd);

	// find minum and maximum
	for (size_t i = 0, minVal_ = maxVal_ = temp[0][0]; i < temp.size(); i++) {
		for (size_t j = 0; j < temp[i].size(); ++j) {
			if (minVal_ > temp[i][j])
				minVal_ = temp[i][j];
			else if (maxVal_ < temp[i][j])
				maxVal_ = temp[i][j];
		}
	}

	// dataholder holds all the data, points only reference it w/ pointers
	dataholder.assign(temp.size(), std::vector<unsigned short>(temp[0].size()));

	for (size_t i = 0; i < temp.size(); ++i) {
		for (size_t j = 0; j < temp[i].size(); ++j) {
			dataholder[i][j] = value2ushort<unsigned short>(temp[i][j]);
		}
	}

	// link points to their data
	datapoints.resize(dataholder.size());
	for (size_t i = 0; i < dataholder.size(); ++i) {
		datapoints[i].data = &dataholder[i];
	}
	bgLog("done\n");
	return true;
}

bool FAMS::importPoints(const multi_img& img) {
	bgLog("Import data points from multispectral image... ");

	// w_ and h_ are only used for result output (i.e. in io.cpp)
	w_ = img.width; h_ = img.height;
	n_ = w_ * h_;
	d_ = (int)img.size(); // dimensionality

	minVal_ = img.minval;
	maxVal_ = img.maxval;

	// let multi_img do the hard work
	dataholder = img.export_ushort(true);

	// link points to their data
	datapoints.resize(dataholder.size());
	for (size_t i = 0; i < dataholder.size(); ++i) {
		datapoints[i].data = &dataholder[i];
	}
	bgLog("done\n");
	return true;
}

cv::Mat1s FAMS::segmentImage() const {
	// mean shift was run on _all_ points
	assert(w_ * h_ == prunedIndex.size());
	cv::Mat1s ret(h_, w_);
	
	cv::Mat1s::iterator it = ret.begin();
	for (int i = 0; it != ret.end(); ++it, ++i) {
		// keep clear of zero
		*it = prunedIndex[i] + 1;
	}
	
	return ret;
}

std::vector<multi_img::Pixel> FAMS::modeVector() const {
	std::vector<multi_img::Pixel> ret(prunedModes.size(), multi_img::Pixel(d_));
	for (size_t i = 0; i < prunedModes.size(); ++i) {
		const std::vector<unsigned short> &src = prunedModes[i];
		multi_img::Pixel &dest = ret[i];
		for (size_t d = 0; d < src.size(); ++d)
			dest[d] = ushort2value(src[d]);
	}
	return ret;
}

void FAMS::saveModes(const std::string& filename, bool pruned) {

	size_t n = (pruned ? prunedModes.size() : modes.size());
	if (n < 1)
		return;

	FILE* fd = fopen((filename).c_str(), "wb");

	for (size_t i = 0; i < n; ++i) {
		std::vector<unsigned short> &src
				= (pruned ? prunedModes[i] : modes[i].data);
		for (size_t d = 0; d < src.size(); ++d) {
			fprintf(fd, "%g ", ushort2value(src[d]));
		}
		fprintf(fd, "\n");
	}

	fclose(fd);
}

void FAMS::saveModeImg(const std::string& filename, bool pruned,
					 const std::vector<multi_img::BandDesc>& ref) {

	size_t n = (pruned ? prunedModes.size() : modes.size());
	if (n < 1)
		return;

	bool full = (n == h_ * w_);
	int h = (full ? h_ : n), w = (full ? w_ : 1);

	multi_img dest(h, w, modes[0].data.size());
	dest.minval = minVal_;
	dest.maxval = maxVal_;
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			multi_img::Pixel px(dest.size());
			std::vector<unsigned short> &src
					= (pruned ? prunedModes[x*y] : modes[x*y].data);
			for (size_t d = 0; d < src.size(); ++d)
				px[d] = ushort2value(src[d]);
			dest.setPixel(y, x, px);
		}
	}

	dest.meta = ref;
	dest.write_out(filename);
}

void FAMS::dbgSavePoints(const std::string& filebase,
						 const std::vector<Point> &points,
						 const std::vector<multi_img::BandDesc>& ref) {
	if (points.size() < 1)
		return;

	multi_img dest((int)points.size(), 1, d_);
	dest.minval = minVal_;
	dest.maxval = maxVal_;
	for (size_t x = 0; x < points.size(); ++x) {
		multi_img::Pixel px(d_);
		for (unsigned int d = 0; d < d_; ++d)
			px[d] = ushort2value((*points[x].data)[d]);
		dest.setPixel(x, 0, px);
	}

	dest.meta = ref;
	dest.write_out(filebase);
}

}
