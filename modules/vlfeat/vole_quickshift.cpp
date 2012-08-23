#include <iostream>
#include <fstream>

#include "vole_quickshift.h"

extern "C" {
#include "vl/quickshift.h"
}

#include "cv.h"
#include "highgui.h"


// if we want an enum for configuration, we need to call this
// ENUM_MAGIC(boundcheck);

namespace vole {

Quickshift::Quickshift(QuickshiftConfig *config, cv::Mat_<cv::Vec3b> img)
	: config(config), img(img)
{
}

Quickshift::~Quickshift() {
}

cv::Mat_<cv::Vec3b> Quickshift::getAverageColoredImage() { return averageColored; }
cv::Mat_<cv::Vec3b> Quickshift::getRandomlyColoredImage() { return randomlyColored; }

std::map<int, std::vector<std::pair<cv::Point, cv::Vec3b> > > &Quickshift::getSegments() {
	return segments;
}

int Quickshift::execute() {
	// TODO: can't that be dangerous in combination with random-dependent methods like IEBV?
	//       using a per commandline option passed random_seed won't reproduce same results anymore.
	srand(static_cast<unsigned int>(time(0) + rand()));

	cv::Mat_<cv::Vec3b> labImg(img.cols, img.rows, cv::Vec3b(0, 0, 0));


	double *qsimg;
	VlQS *qs;
	if (config->use_chroma) {
		qsimg = new double[img.cols * img.rows * 2];
		for (int y = 0; y < img.rows; ++y) {
			for (int x = 0; x < img.cols; ++x) {
				double sum = 0;
				for (int i = 0; i < 3; ++i) sum += img[y][x][i];
				if (sum < 0.0001) sum = 1;

				double chromaR = static_cast<double>(img[y][x][2]) / sum;
				double chromaB = static_cast<double>(img[y][x][0]) / sum;
				// offenbar spaltenweise, farbkanaele hintereinander...
				qsimg[x*img.rows + y + 0*img.cols*img.rows] = chromaR;
				qsimg[x*img.rows + y + 1*img.cols*img.rows] = chromaB;
			}
		}
		qs = ::vl_quickshift_new(qsimg, img.rows, img.cols, 2);
	} else {
		cvtColor(img, labImg, CV_BGR2Lab);
		qsimg = new double[labImg.cols * labImg.rows * 3];
		for (int y = 0; y < labImg.rows; ++y) {
			for (int x = 0; x < labImg.cols; ++x) {
				for (int c = 0; c < 3; ++c) {
					// offenbar spaltenweise, farbkanaele hintereinander...
					qsimg[x*labImg.rows + y + c*labImg.cols*labImg.rows]
						= static_cast<double>(labImg[y][x][c]) / 255.0;
				}
			}
		}
		qs = ::vl_quickshift_new(qsimg, labImg.rows, labImg.cols, 3);
	}

	

	if (config->kernel_size > 0) {
		vl_quickshift_set_kernel_size(qs, config->kernel_size);
		if (config->max_dist_multiplier > 0) {
			double max_dist = config->kernel_size * config->max_dist_multiplier;
			vl_quickshift_set_max_dist(qs, max_dist);
		}
	}

	if (config->verbosity > 0) {
		std::cout << "kernel_size = " << ::vl_quickshift_get_kernel_size(qs) << std::endl;
		std::cout << "max_dist = " << ::vl_quickshift_get_max_dist(qs) << std::endl;
	}

	vl_quickshift_process(qs);

	int *par = ::vl_quickshift_get_parents(qs);
	double *dists = ::vl_quickshift_get_dists(qs);
	int nElems = img.rows * img.cols;

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
			std::cerr << "ups, elem " << i << " (dist = "
				<< dists[i] << " fuehrt zu pos " << pos << ")" << std::endl;
			continue;
		}
		hist[pos]++;
	}
	if (config->verbosity > 0) {
		std::cout << "minD = " << minD << ", maxD = " << maxD << std::endl;
		for (int i = 0; i < 20; ++i) {
			std::cout << " " << hist[i];
		}
		std::cout << std::endl;
	}

	double distLimit = (maxD - minD) * config->merge_threshold;

	if (config->verbosity > 0) {
		std::cout << "dist limit: " << distLimit << std::endl;
	}

	for (int i = 0; i < nElems; ++i) {
		int cur = i;
		while ((dists[cur] < distLimit) && (par[cur] != cur)) {
			if ((par[par[cur]] != par[cur]) && (dists[par[cur]] < distLimit))
				par[cur] = par[par[cur]];
			cur = par[cur];
		}
	}

	// collect a list of all segments
	segments.clear();
	for (int i = 0; i < nElems; ++i) {
		if (segments.find(par[i]) == segments.end()) { // insert new segment
			std::vector<std::pair<cv::Point, cv::Vec3b> > tmp;
			segments[par[i]] = tmp;
		}
		int x = i / img.rows;
		int y = i % img.rows;
		cv::Point idx(x, y);
		if (config->use_chroma) {
			segments[par[i]].push_back(std::pair<cv::Point, cv::Vec3b>(
				idx, cv::Vec3b(img[y][x][0], img[y][x][1], img[y][x][2]))
			);
		} else {
			segments[par[i]].push_back(std::pair<cv::Point, cv::Vec3b>(idx, labImg[y][x]));
		}
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
			if ((img[y][x][0] == 0) && (img[y][x][1] == 0) && (img[y][x][2] == 0)) continue;
			avg[0] = avg[0] + cur_seg[i].second[0];
			avg[1] = avg[1] + cur_seg[i].second[1];
			avg[2] = avg[2] + cur_seg[i].second[2];
			count++;
		}
		if (count == 0) count = 1;
		avg[0] = avg[0] / count;
		avg[1] = avg[1] / count;
		avg[2] = avg[2] / count;
		avg_cols.insert(std::pair<int, cv::Vec3b>(it->first, cv::Vec3b((uchar)avg[0], (uchar)avg[1], (uchar)avg[2])));
	}

	cv::Mat_<cv::Vec3b> outputLab(img.rows, img.cols, cv::Vec3b(0, 0, 0));
	cv::Mat_<cv::Vec3b> outputLab2(img.rows, img.cols, cv::Vec3b(0, 0, 0));
	for (it = segments.begin(); it != segments.end(); ++it) {
		cv::Vec3b col = avg_cols[it->first];
		cv::Vec3b col2(abs(rand()) % 255, abs(rand()) % 255, abs(rand()) % 255);
		std::vector<std::pair<cv::Point, cv::Vec3b> > &cur_seg = it->second;
		for (unsigned int i = 0; i < cur_seg.size(); ++i) {
			int x = cur_seg[i].first.x;
			int y = cur_seg[i].first.y;
			if ((img[y][x][0] == 0) && (img[y][x][1] == 0) && (img[y][x][2] == 0)) continue;
			outputLab[y][x] = col;
			outputLab2[y][x] = col2;
		}
	}
	averageColored = cv::Mat_<cv::Vec3b>(outputLab.rows, outputLab.cols, cv::Vec3b(0, 0, 0));
	randomlyColored = cv::Mat_<cv::Vec3b>(outputLab2.rows, outputLab2.cols, cv::Vec3b(0, 0, 0));
	cvtColor(outputLab, averageColored, CV_Lab2BGR);
	cvtColor(outputLab2, randomlyColored, CV_Lab2BGR);
	cv::Vec3b nullVec(0, 0, 0);
	for (int y = 0; y < img.rows; ++y) {
		for (int x = 0; x < img.cols; ++x) {
			if ((img[y][x][0] == 0) && (img[y][x][1] == 0) && (img[y][x][2] == 0)) {
				averageColored[y][x] = nullVec;
				randomlyColored[y][x] = nullVec;
			}
		}
	}
	::vl_quickshift_delete(qs);
	delete[] qsimg;

	return 0;
}

} // namespace vole
