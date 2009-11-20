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

multi_img multi_img::clone() {
	multi_img ret(size());
	for (int i = 0; i < size(); ++i)
		ret[i] = (*this)[i].clone();

	return ret;
}

unsigned short* multi_img::export_interleaved() const {
	double scale = 65535.0/(maxval - minval);
	unsigned short *ret = new unsigned short[size()*width*height];
	
	/* actually we don't care about the ordering of the pixels, just all
	   values have to get in there in interleaved format */
	cv::MatConstIterator_<double> it;
	register int d, i;
	for (d = 0; d < size(); ++d) {
		const cv::Mat_<double> &img = (*this)[d];
		// i starts with offset d, then jumps over dimensions
		for (it = img.begin(), i = d; it != img.end(); ++it, i+=size())
			ret[i] = (unsigned short)((*it - minval) * scale);
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
	/* our favorite range */
	minval = 0.; maxval = 255.;

	for (int fi = 0; fi < files.size(); ++fi) {
	    cv::Mat src = cv::imread(files[fi].c_str(), -1); // flag -1: preserve format
	    
		if (src.empty()) {
			cerr << "ERROR: Failed to load " << files[fi] << endl;
			continue;
		}
		
		// write or test size
		if (width > 0) {
			if (src.cols != width || src.rows != height) {
				cerr << "ERROR: Size mismatch for image " << files[fi] << endl;
				continue;
			}
		} else {
			width = src.cols;
			height = src.rows;
		}
		
		// we only expect CV_8U or CV_16U right now
		double srcmaxval = (src.depth() == CV_16U ? 65535. : 255.);

		// operate on sane data type from now on
		cv::Mat_<double> img = src;
		// rescale data accordingly
		if (maxval != srcmaxval) // evil comp., doesn't hurt
			img *= maxval/srcmaxval;
		
		// split image
		std::vector<cv::Mat> channels;
		cv::split(img, channels);
				
		// add everything in at the end
		insert(end(), channels.begin(), channels.end());
		
	    cout << "Added " << files[fi] << ":\t" << channels.size()
             << (channels.size() == 1 ? " channel, " : " channels, ")
             << (src.depth() == CV_16U ? 16 : 8) << " bits" << endl;
	}

	cout << "Total of " << size() << " dimensions" << endl;
}

void multi_img::write_out(const string& base, bool normalize) {
	char name[1024];
	for (int i = 0; i < size(); ++i) {
		sprintf(name, "%s%02d.png", base.c_str(), i);

		if (normalize) {
			cv::Mat normalized;
			double scale = 255./(maxval - minval);
			(*this)[i].convertTo(normalized, CV_8U, scale, -scale*minval);
			cv::imwrite(name, normalized);
		} else
			cv::imwrite(name, (*this)[i]);
	}
}

void multi_img::apply_logarithm() {
	for (int i = 0; i < size(); ++i) {
		// will assign large negative value to 0 pixels
		cv::log((*this)[i], (*this)[i]);
		// get rid of negative values (when pixel value was 0)
		cv::max((*this)[i], 0., (*this)[i]);
	}
	// data format has changed as follows
	minval = 0.;
	maxval = log(maxval);
}

multi_img multi_img::spec_gradient() {
	multi_img ret;
	// data format of output
	ret.minval = -maxval/2.;
	ret.maxval =  maxval/2.;
	ret.width = width; ret.height = height;

	for (int i = 0; i < size()-1; ++i) {
		ret.push_back((*this)[i+1] - (*this)[i]);
	}
	return ret;
}
