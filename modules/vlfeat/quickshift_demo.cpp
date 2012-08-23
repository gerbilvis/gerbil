#include <iostream>
#include <fstream>
#include <cstdlib>
#include <limits>

#include "quickshift_demo.h"

extern "C" {
#include "vl/quickshift.h"
}

#include "cv.h"
#include "highgui.h"

using namespace boost::program_options;

// if we want an enum for configuration, we need to call this
// ENUM_MAGIC(boundcheck);

quickshiftDemo::quickshiftDemo()
 : Command(
		"qs_demo",
		config,
		"Christian Riess",
		"christian.riess@informatik.uni-erlangen.de")
{
}

quickshiftDemo::~quickshiftDemo() {
}



int quickshiftDemo::execute() {
	if (config.verbosity > 0) std::cout << "IEBV quickshift Versuch" << std::endl;
	// TODO: can't that be dangerous in combination with random-dependent methods like IEBV?
	//       using a per commandline option passed random_seed won't reproduce same results anymore.
	srand(static_cast<unsigned int>(time(0) + rand()));

	if (config.input_file.length() < 1) {
		std::cerr << "No input file given (add -I <input_file>)!" << std::endl;
		printHelp();
		return 1;
	}
	cv::Mat_<cv::Vec3b> img = cv::imread(config.input_file);
	cv::Mat_<cv::Vec3b> labImg(img.cols, img.rows, cv::Vec3b(0, 0, 0));

	cvtColor(img, labImg, CV_BGR2Lab);

	double *qsimg = new double[labImg.cols * labImg.rows * 3];
	for (int y = 0; y < labImg.rows; ++y) {
		for (int x = 0; x < labImg.cols; ++x) {
			for (int c = 0; c < 3; ++c) {
//				denkste...
//				qsimg[(y*labImg.cols + x)*3 + c] = static_cast<double>(labImg[y][x][c]) / 255.0;
				// offenbar spaltenweise, farbkanaele hintereinander...
				qsimg[x*labImg.rows + y + c*labImg.cols*labImg.rows] = static_cast<double>(labImg[y][x][c]) / 255.0;
//				qsimg[y + x*labImg.rows + c*labImg.cols*labImg.rows] = static_cast<double>(labImg[y][x][c]);
			}
		}
	}

	std::cout << "rows = " << labImg.rows << " cols = " << labImg.cols << std::endl;

	VlQS *qs = ::vl_quickshift_new(qsimg, labImg.rows, labImg.cols, 3);

	double max_dist = config.kernel_size*config.max_dist_multiplier;

	std::cout << "kernel_size = " << ::vl_quickshift_get_kernel_size(qs);
	std::cout << "max_dist = " << ::vl_quickshift_get_max_dist(qs);

	::vl_quickshift_set_kernel_size(qs, config.kernel_size);
	::vl_quickshift_set_max_dist(qs, max_dist);
	std::cout << "vor process..." << std::endl;
	::vl_quickshift_process(qs);
	int *par = ::vl_quickshift_get_parents(qs);
	double *dists = ::vl_quickshift_get_dists(qs);
	int nElems = labImg.rows * labImg.cols;

	double minD = DBL_MAX;
	double maxD = DBL_MIN;
	for (int i = 0; i < nElems; ++i) {
		if (std::numeric_limits<double>::infinity() == dists[i]) continue;
		if (dists[i] < minD) minD = dists[i];
		if (dists[i] > maxD) maxD = dists[i];
	}

	int hist[20];
	for (int i = 0; i < 20; ++i) hist[i] = 0;

	for (int i = 0; i < nElems; ++i) {
		if (std::numeric_limits<double>::infinity() == dists[i]) continue;
		unsigned int pos = static_cast<unsigned int>(20.0*(dists[i]-minD)/(maxD-minD));
		if (pos == 20) pos = 19;
		if ((pos < 0) || (pos >= 20)) {
			std::cerr << "ups, elem " << i << " (dist = " << dists[i] << " fuehrt zu pos " << pos << std::endl;
			continue;
		}
		hist[pos]++;
	}
	std::cout << "minD = " << minD << ", maxD = " << maxD << std::endl;
	for (int i = 0; i < 20; ++i) {
		std::cout << " " << hist[i];
	}
	std::cout << std::endl;

	double distLimit = (maxD - minD) * config.merge_threshold;
	std::cout << "dist limit: " << distLimit << std::endl;

	std::cout << "coloring the parent..." << std::endl;

	for (int i = 0; i < nElems; ++i) {
		int cur = i;
		while ((dists[cur] < distLimit) && (par[cur] != cur)) {
			if ((par[par[cur]] != par[cur]) && (dists[par[cur]] < distLimit))
				par[cur] = par[par[cur]];
			cur = par[cur];
		}
	}

	// collect a list of all segments
	std::map<int, std::vector<std::pair<cv::Point, cv::Vec3b> > > segments;
	for (int i = 0; i < nElems; ++i) {
		if (segments.find(par[i]) == segments.end()) { // insert new segment
			std::vector<std::pair<cv::Point, cv::Vec3b> > tmp;
			segments[par[i]] = tmp;
		}
		int x = i / labImg.rows;
		int y = i % labImg.rows;
		cv::Point idx(x, y);
		segments[par[i]].push_back(std::pair<cv::Point, cv::Vec3b>(idx, labImg[y][x]));
	}

	std::map<int, cv::Vec3b> avg_cols;

	std::map<int, std::vector<std::pair<cv::Point, cv::Vec3b> > >::iterator it;
	for (it = segments.begin(); it != segments.end(); ++it) {
		cv::Vec3d avg(0, 0, 0);
		std::vector<std::pair<cv::Point, cv::Vec3b> > &cur_seg = it->second;
		int count = 0;
		for (unsigned int i = 0; i < cur_seg.size(); ++i) {
			int x = cur_seg[i].first.x;
			int y = cur_seg[i].first.y;
			if ((img[y][x][0] == 0) && (img[y][x][0] == 0) && (img[y][x][0] == 0)) continue;
			avg[0] = avg[0] + cur_seg[i].second[0];
			avg[1] = avg[1] + cur_seg[i].second[1];
			avg[2] = avg[2] + cur_seg[i].second[2];
			count++;
		}
		avg[0] = avg[0] / count;
		avg[1] = avg[1] / count;
		avg[2] = avg[2] / count;
		avg_cols[it->first] = cv::Vec3b((uchar)avg[0], (uchar)avg[1], (uchar)avg[2]);
	}

	cv::Mat_<cv::Vec3b> outputLab(labImg.rows, labImg.cols, cv::Vec3b(0, 0, 0));
	cv::Mat_<cv::Vec3b> outputLab2(labImg.rows, labImg.cols, cv::Vec3b(0, 0, 0));
	for (it = segments.begin(); it != segments.end(); ++it) {
		cv::Vec3b col = avg_cols[it->first];
		cv::Vec3b col2(abs(rand()) % 255, abs(rand()) % 255, abs(rand()) % 255);
		std::vector<std::pair<cv::Point, cv::Vec3b> > &cur_seg = it->second;
		for (unsigned int i = 0; i < cur_seg.size(); ++i) {
			int x = cur_seg[i].first.x;
			int y = cur_seg[i].first.y;
			if ((img[y][x][0] == 0) && (img[y][x][0] == 0) && (img[y][x][0] == 0)) continue;
			outputLab[y][x] = col;
			outputLab2[y][x] = col2;
		}
	}
	cv::Mat_<cv::Vec3b> output(outputLab.rows, outputLab.cols, cv::Vec3b(0, 0, 0));
	cv::Mat_<cv::Vec3b> output2(outputLab2.rows, outputLab2.cols, cv::Vec3b(0, 0, 0));
	cvtColor(outputLab, output, CV_Lab2BGR);
	cvtColor(outputLab2, output2, CV_Lab2BGR);
	cv::Vec3b nullVec(0, 0, 0);
	for (int y = 0; y < img.rows; ++y) {
		for (int x = 0; x < img.cols; ++x) {
			if ((img[y][x][0] == 0) && (img[y][x][0] == 0) && (img[y][x][0] == 0)) {
				output[y][x] = nullVec;
				output2[y][x] = nullVec;
			}
		}
	}
	cv::imwrite(config.output_file, output);

//	for (int i = 0; i < nElems; ++i) {
//		if (par[i] == i) { // self-root?
//			output2[i % img.rows][i / img.rows] = 0;
//		} else {
//			output2[i % img.rows][i / img.rows] = (255/8.0)*((par[i] % 7)+1);
//		}
//	}
	cv::imwrite("/tmp/somesegments.png", output2);
	::vl_quickshift_delete(qs);


	

	return 0;
}


void quickshiftDemo::printShortHelp() const {
	std::cout << "frontend for quickshift from the vlfeat lib" << std::endl;
}


void quickshiftDemo::printHelp() const {
	std::cout << "frontend for quickshift from the vlfeat lib" << std::endl;
	std::cout << std::endl;
	std::cout << "applies quickshift - how this is done is not clear. Please refer" << std::endl
	 << "A. Vedaldi and S. Soatto. \"Quick Shift and Kernel Methods for Mode Seeking\", in Proc. ECCV, 2008."
		<< std::endl << std::endl;
}

