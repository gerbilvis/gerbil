#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>
#include <cv.h>
#include <highgui.h>
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
			float r = pttemp[3*i]/255., g = pttemp[3*i+1]/255., b = pttemp[3*i+2]/255.;
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
	rr_        = new double[d_]; // TODO: get rid of!

	for (i = 0; i < length; i++)
		data_[i] = (unsigned short)(65535.0 * (pttemp[i] - minVal_) / deltaVal);
	delete [] pttemp;

	points_ = new fams_point[n_];
	unsigned short *dtempp;
	for (i = 0, dtempp = data_; i < n_; i++, dtempp += d_) {
		points_[i].data_     = dtempp;
		points_[i].usedFlag_ = 0;
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
	d_ = img.size(); // dimensionality

	minVal_ = img.minval; // TODO: get rid of!
	maxVal_ = img.maxval;

	// setup storage
	dataSize_ = d_ * sizeof(unsigned short); // size of one point
	rr_ = new double[d_]; // TODO: get rid of!

	// let multi_img do the hard work
	data_ = img.export_interleaved();

	// assign points to data chunks
	points_ = new fams_point[n_];
	int i;
	unsigned short *loc;
	for (i = 0, loc = data_; i < n_; i++, loc += d_) {
		points_[i].data_     = loc;
		points_[i].usedFlag_ = 0;
	}
	bgLog("done\n");
	return true;
}

void FAMS::SaveMymodes(const std::string& filebase) {
	FILE* fp = fopen((filebase + ".modes.my.txt").c_str(), "wb");

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

IplImage* FAMS::segmentImage(bool normalize) {
	// mean shift was run on _all_ points
	assert(w_ * h_ == nsel_);

	IplImage *ret = cvCreateImage(cvSize(w_, h_), IPL_DEPTH_8U, 1);
	register int i = 0;
	register unsigned char *row;
	unsigned char maxval = 0;
	for (int y = 0; y < h_; ++y) {
		row = (unsigned char*)(ret->imageData + ret->widthStep*y);
		for (int x = 0; x < w_; ++x) {
			row[x] = __min(indmymodes[i], 255);
			maxval = __max(maxval, row[x]);
			++i;
		}
	}
	if (normalize)
		cvScale(ret, ret, 255./(double)maxval, 0.);
	
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


void FAMS::SaveModes(const std::string& filebase) {
	if (nsel_ < 1)
		return;

	FILE* fd = fopen((filebase + ".modes.txt").c_str(), "wb");
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

void FAMS::SavePrunedModes(const std::string& filebase) {
	if (npm_ < 1)
		return;
	FILE* fd = fopen((filebase + ".modes.joined.txt").c_str(), "wb");
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


int FAMS::LoadBandwidths(const char* fn) {
	FILE* fd;
	fd = fopen(fn, "rb");
	if (fd == NULL)
		return 0;
	int n, i;
	fscanf(fd, "%d", &n);
	if (n != n_) {
		fclose(fd);
		return 0;
	}
	float bw;
	float deltaVal = maxVal_ - minVal_;
	for (i = 0; i < n_; i++) {
		fscanf(fd, "%g", &bw);
		points_[i].window_ = (unsigned int)(65535.0 * (bw) / deltaVal);
	}
	fclose(fd);
	return 1;
}

void FAMS::SaveBandwidths(const char* fn) {
	FILE* fd;
	fd = fopen(fn, "wb");
	if (fd == NULL)
		return;
	fprintf(fd, "%d\n", n_);
	float bw;
	float deltaVal = maxVal_ - minVal_;
	int   i;
	for (i = 0; i < n_; i++) {
		bw = (float)(points_[i].window_ * deltaVal / 65535.0);
		fprintf(fd, "%g\n", bw);
	}
	fclose(fd);
}

