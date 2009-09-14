#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <highgui.h>
#include "multi_img.h"

using namespace std;


multi_img::multi_img(const string& filename) : width(0), height(0) {
	vector<string> files;
	
	// try to read in file list first
	files = read_filelist(filename);
	if (files.empty()) {
		// maybe we got a single image as argument
		files.push_back(filename);
	}
	
	// read in image data
	read_image(files);
};

void multi_img::cleanup() {
	for (int i = 0; i < size(); ++i)
		cvReleaseImage(&(*this)[i]);
	clear();
}

multi_img multi_img::clone() {
	multi_img ret(size());
	for (int i = 0; i < size(); ++i)
		ret[i] = cvCloneImage((*this)[i]);

	return ret;
}

unsigned short* multi_img::export_interleaved() const {
	IplImage *img;
	double scale = 65535.0/(maxval - minval);
	unsigned short *ret = new unsigned short[size()*width*height];
	
	/* actually we don't care about the ordering of the pixels, just all
	   values have to get in there in interleaved format */
	int x, y, d;
	register int i;
	register double *row;
	for (d = 0; d < size(); ++d) {
		i = d; // start with right offset
		img = (*this)[d];
		for (y = 0; y < height; ++y) {
			row = (double*)(img->imageData + img->widthStep*y);
			for (x = 0; x < width; ++x) {
				ret[i + size()*x] = ((unsigned short)row[x] - minval)*scale;
			}
			// every row consists of width*dim values
			i += width*size();
		}
	}
	
	return ret;
}

// parse file list
vector<string> multi_img::read_filelist(const string& filename) {
	vector<string> ret;
	FILE *in = fopen(filename.c_str(), "r");
	if (!in)
		return ret;
	
	int count;
	char tmp[1024];
	string base;
	if (fscanf(in, "%d %s", &count, tmp) != 2)
		return ret;

	base = tmp; base.append("/");
	for (; count > 0; count--) {
		if (fscanf(in, "%s", tmp) != 1)
			return vector<string>(); // file inconsistent -> screw it!
		ret.push_back(base + tmp);
	}
	fclose(in);
	return ret;
}

// read multires. image into vector
void multi_img::read_image(vector<string>& files) {
	CvSize cvsize; // for convenience
	
	for (int fi = 0; fi < files.size(); ++fi) {
	    IplImage *src = cvLoadImage(files[fi].c_str(),
	                           CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	    
		if (!src) {
			cerr << "ERROR: Failed to load " << files[fi] << endl;
			continue;
		}
		
		// write or test size
		if (width > 0) {
			if (src->width != width || src->height != height) {
				cerr << "ERROR: Size mismatch for image " << files[fi] << endl;
				continue;
			}
		} else {
			width = src->width;
			height = src->height;
			cvsize = cvSize(width, height);
		}
		
		// convert to float to make life easier
		// also split channels to images if applicable
		IplImage *tmpf[src->nChannels];
		
		/* for a scaling to 0..255 */
		double scale = 1./(double)(1 << (src->depth - 8));
		double shift = 0.;
		minval = 0.; maxval = 255.;

		if (src->nChannels == 1) {
			tmpf[0] = cvCreateImage(cvsize, IPL_DEPTH_64F, 1);
			cvConvertScale(src, tmpf[0], scale, shift);
		} else {
			IplImage *tmp[src->nChannels];
			for (int i = 0; i < src->nChannels; ++i)
				tmp[i] = cvCreateImage(cvsize, src->depth, 1);
			cvSplit(src, tmp[0], tmp[1], tmp[2], NULL);
			for (int i = 0; i < src->nChannels; ++i) {
				tmpf[i] = cvCreateImage(cvsize, IPL_DEPTH_64F, 1);
				cvConvertScale(tmp[i], tmpf[i], scale, shift);
				cvReleaseImage(&tmp[i]);
			}
		}
		
		// add everything in
		for (int i = 0; i < src->nChannels; ++i)
			push_back(tmpf[i]);
	    cout << "Added " << files[fi] << ":\t" << src->nChannels << " channels, "
	              << src->depth  << " bits" << endl;
	    
	    cvReleaseImage(&src);
	}

	cout << "Total of " << size() << " dimensions" << endl;
}

void multi_img::write_out(const string& base, bool normalize) {
	char name[1024];
	double scale = 255./(maxval - minval);
	for (int i = 0; i < size(); ++i) {
		sprintf(name, "%s%02d.png", base.c_str(), i);

		/* we have it in scale 0..255 */
		if (normalize) {
			IplImage *urgs = cvCreateImage(cvSize(width, height), IPL_DEPTH_64F, 1);
			cvConvertScale((*this)[i], urgs, scale, -scale*minval);
			cvSaveImage(name, urgs);
			cvReleaseImage(&urgs);
		} else
			cvSaveImage(name, (*this)[i]);
	}
}

void multi_img::apply_logarithm() {
	for (int i = 0; i < size(); ++i) {
		// will assign large negative value to 0 pixels
		cvLog((*this)[i], (*this)[i]);
		// get rid of negative values (when pixel value was 0)
		cvMaxS((*this)[i], 0., (*this)[i]);
	}
	// data format has changed as follows
	minval = 0.;
	maxval = log(maxval);
}

multi_img multi_img::spec_gradient() {
	CvSize cvsize = cvSize(width, height); // letter case is always fun
	multi_img ret;
	// data format of output
	ret.minval = -maxval/2.;
	ret.maxval = maxval/2.;
	ret.width = width; ret.height=height;

	for (int i = 0; i < size()-1; ++i) {
		IplImage *tmp = cvCreateImage(cvsize, IPL_DEPTH_64F, 1);
		cvSub((*this)[i+1], (*this)[i], tmp);
		ret.push_back(tmp);		
	}
	return ret;
}
