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

bool FAMS::LoadPoints(char* filename) {
	bgLog("Load data points from P3 NetPBM file %s... ", filename);
	CleanPoints();
	FILE * fd;
	char head[255];
	int  bpp;
	fd = fopen(filename, "r");

	if (!fd) {
		bgLog("Error opening %s\n", filename);
		return false;
	}
	if ((fscanf(fd, "%s %d %d %d", head, &w_, &h_, &bpp) == 4)
		&& (strcmp(head, "P3") == 0) && (bpp == 255)) {
		n_ = w_ * h_;
		d_ = 3;
	} else {
		bgLog("Error reading %s\n", filename);
		fclose(fd);
		return false;
	}

	// for convenience
	int length = n_ * d_;
	
	// allocate data
	float* pttemp = new float[length];
	int i;
	for (i = 0; i < length; ++i) {
		if (fscanf(fd, "%f", &pttemp[i]) != 1) {
			bgLog("Error reading %s\n", filename);
			fclose(fd);
			delete [] pttemp;
			return false;
		}
	}
	fclose(fd);

	/* cut out semi white pixels, yay!
	   (good if you have white background with JPG artifacts at object borders)
	
		float hsv[3];
		for (i=0; i<n_; ++i) {
			float r = pttemp[3*i]/255., g = pttemp[3*i+1]/255.,
				  b = pttemp[3*i+2]/255.;
			rgb2hsv(r, g, b, &hsv[0],&hsv[1],&hsv[2]);
			if (hsv[1] < 0.08 && hsv[2] > 0.88)
				pttemp[3*i] = pttemp[3*i+1] = pttemp[3*i+2] = 255.;
		//		printf("RGB: %3g %3g %3g \t HSV: %3g %3g %3g \n",
		//			pttemp[3*i],pttemp[3*i+1],pttemp[3*i+2], hsv[0],hsv[1],hsv[2]);
		}
	 */

	// convert from RGB to LUV ********************************
	//	rgb2luv(pttemp, pttemp, n_*d_);

	// allocate and convert to integer
	for (i = 0, minVal_ = maxVal_ = pttemp[0]; i < length; i++) {
		if (minVal_ > pttemp[i])
			minVal_ = pttemp[i];
		else if (maxVal_ < pttemp[i])
			maxVal_ = pttemp[i];
	}
	double deltaVal = maxVal_ - minVal_;
	if (deltaVal == 0)
		deltaVal = 1;

	hasPoints_ = 1;
	dataSize_ = d_ * sizeof(unsigned short);

	// data_ holds all the data, points only reference it w/ pointers
	data_      = new unsigned short[n_ * d_];

	for (i = 0; i < length; i++)
		data_[i] = (unsigned short)(65535.0 * (pttemp[i] - minVal_) / deltaVal);
	delete [] pttemp;

	points_ = new fams_point[n_];
	unsigned short *dtempp;
	for (i = 0, dtempp = data_; i < n_; i++, dtempp += d_) {
		points_[i].data_     = dtempp;
	}
	bgLog("done\n");
	return true;
}

bool FAMS::ImportPoints(const multi_img& img) {
	bgLog("Import data points from multispectral image... ");
	// set member variables
	CleanPoints();
	hasPoints_ = 1;
	// w_ and h_ are only used for result output (i.e. in io.cpp)
	w_ = img.width; h_ = img.height;
	n_ = w_ * h_;
	d_ = (int)img.size(); // dimensionality

	minVal_ = img.minval; // TODO: get rid of!
	maxVal_ = img.maxval;

	// setup storage
	dataSize_ = d_ * sizeof(unsigned short); // size of one point

	// let multi_img do the hard work
	data_ = img.export_interleaved(true);

	// assign points to data chunks
	points_ = new fams_point[n_];
	int i;
	unsigned short *loc;
	for (i = 0, loc = data_; i < n_; i++, loc += d_) {
		points_[i].data_     = loc;
	}
	bgLog("done\n");
	return true;
}

void FAMS::SaveMymodes(const std::string& filebase) {
	FILE* fp = fopen((filebase + "modes.my.txt").c_str(), "wb");

	int i, j, idx;
	idx = 0;

	for (i = 0; i < nsel_; i++) {
		for (j = 0; j < d_; j++) {
			fprintf(fp, "%g ",
					testmymodes[idx] * (maxVal_ - minVal_) / 65535.0 + minVal_);

			idx++;
		}
		fprintf(fp, "\n");
	}

	fclose(fp);
}

cv::Mat1s FAMS::segmentImage() {
	// mean shift was run on _all_ points
	assert(w_ * h_ == nsel_);
	cv::Mat1s ret(h_, w_);
	
	cv::Mat1s::iterator it = ret.begin();
	for (int i = 0; it != ret.end(); ++it, ++i) {
		// keep clear of zero
		*it = indmymodes[i] + 1;
	}
	
	return ret;
}


void FAMS::CreatePpm(char *fn) { // TODO: get rid of! export as multiimg instead
	std::string filename(fn);

	float out[3];

//	filename = fn + ".txt";
	ifstream in((filename + ".txt").c_str(), ios::in);
	if (!in) {
		cout << "File %s does not exist" << filename;
		exit(1);
	}
	ofstream pgmf((filename + ".ppm").c_str());
	if (!pgmf) {
		cout << "File %s can not be opened for writing" << fn << endl;
	}

	pgmf << "P3" << endl << w_ << " " << h_ << " " << "255" << endl;

	for (int i = 1; i <= w_ * h_; ++i) {
		in >> out[0];
		in >> out[1];
		in >> out[2];
		// convert from LUV to RGB ********************************
		//	luv2rgb(out, out, 1);
		pgmf << (int)out[0] << " " << (int)out[1] << " " << (int)out[2];
		if (i % 6)
			pgmf << " ";
		else
			pgmf << endl;
	}

	bgLog("Image is stored in  %s\n", (filename + ".ppm").c_str());

	in.close();
	pgmf.close();
}

void FAMS::SaveModeImg(const std::string& filebase,
					 const std::vector<multi_img::BandDesc>& ref) {
	if (nsel_ < 1)
		return;

	bool full = (nsel_ == h_ * w_);

	int h = (full? h_ : nsel_), w = (full? w_ : 1);
	multi_img dest(h, w, d_);
	dest.minval = minVal_;
	dest.maxval = maxVal_;
	int idx = 0;
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			multi_img::Pixel px(d_);
			for (int d = 0; d < d_; ++d)
				px[d] = (multi_img::Value)modes_[idx++] * (maxVal_ - minVal_)
						/ 65535.0f + minVal_;
			dest.setPixel(y, x, px);
		}
	}

	dest.meta = ref;
	dest.write_out(filebase);
}

void FAMS::DbgSavePoints(const std::string& filebase,
						 const std::vector<fams_point> points,
						 const std::vector<multi_img::BandDesc>& ref) {
	if (points.size() < 1)
		return;

	multi_img dest((int)points.size(), 1, d_);
	dest.minval = minVal_;
	dest.maxval = maxVal_;
	for (int x = 0; x < points.size(); ++x) {
		multi_img::Pixel px(d_);
		for (int d = 0; d < d_; ++d)
			px[d] = (multi_img::Value)points[x].data_[d] * (maxVal_ - minVal_)
					/ 65535.0f + minVal_;
		dest.setPixel(x, 0, px);
	}

	dest.meta = ref;
	dest.write_out(filebase);
}

void FAMS::SaveModes(const std::string& filebase) {
	if (nsel_ < 1)
		return;

	FILE* fd = fopen((filebase + "modes.txt").c_str(), "wb");
	int i, j, idx;
	idx = 0;
	for (i = 0; i < nsel_; i++) {
		for (j = 0; j < d_; j++) {
			fprintf(fd, "%g ",
					modes_[idx++] * (maxVal_ - minVal_) / 65535.0 + minVal_);
		}
		fprintf(fd, "\n");
	}
	fclose(fd);
	bgLog("done\n");
}

void FAMS::SavePrunedModeImg(const std::string& filebase,
					 const std::vector<multi_img::BandDesc>& ref) {
	if (npm_ < 1)
		return;

	bool full = false;

	int h = (full? h_ : npm_), w = (full? w_ : 1);
	multi_img dest(h, w, d_);
	dest.minval = minVal_;
	dest.maxval = maxVal_;
	int idx = 0;
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			multi_img::Pixel px(d_);
			for (int d = 0; d < d_; ++d)
				px[d] = (multi_img::Value)prunedmodes_[idx++] * (maxVal_ - minVal_)
						/ 65535.0f + minVal_;
			dest.setPixel(y, x, px);
		}
	}

	dest.meta = ref;
	dest.write_out(filebase);
}

void FAMS::SavePrunedModes(const std::string& filebase) {
	if (npm_ < 1)
		return;
	FILE* fd = fopen((filebase + "modes.joined.txt").c_str(), "wb");
	int i, j, idx;
	idx = 0;
	for (i = 0; i < npm_; i++) {
		fprintf(fd, "%d  ", nprunedmodes_[i]);
		for (j = 0; j < d_; j++) {
			fprintf(
				fd, "%g ", prunedmodes_[idx++] *
				(maxVal_ - minVal_) / 65535.0 + minVal_);
		}
		fprintf(fd, "\n");
	}
	fclose(fd);
	bgLog("done\n");
}
