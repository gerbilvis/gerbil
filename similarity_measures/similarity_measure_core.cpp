/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>,
	Felix Lugauer, Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

#include "similarity_measure_core.h"

#include "data_traits.h"

#include "normalized_cross_correlation.h"
#include "correlation_coefficient_histogram.h"
#include "normalized_mutual_information.h"
#include "mutual_information_histogram.h"
#include "mean_reciprocal_square_difference.h"
#include "mean_squares.h"
#include "gradient_difference.h"
#include "mean_squares_histogram.h"
#include "earth_movers_distance.h"


// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace vole {

SimilarityMeasureCore::SimilarityMeasureCore() {
	initClass();
}

SimilarityMeasureCore::SimilarityMeasureCore(SimilarityMeasureConfig *config)
	: config(config)
{

	initClass();
	setConfig(config);

	/*
	std::cout << "#  Trying to read in template banknote first...  #"	 << std::endl;
	banknoteTemplate = new vise::Banknote(inputdir.c_str(),templateNr);
	std::cout << "#  I/O done! Now generate fused banknote images...  #"	 << std::endl;
	banknoteTemplate->generateCombinedNotes(a,b);

	std::cout << "#  Trying to read banknote which will be compared to the template...  #"	 << std::endl;
	banknote = new vise::Banknote(inputdir.c_str(),banknoteNr);
	std::cout << "#  I/O done! Now generate fused banknote images...  #"	 << std::endl;
	banknote->generateCombinedNotes(a,b);

*/
}

void SimilarityMeasureCore::initClass() {
	metric_names[0] = "ms";   metrics[0] = MS;   cmpObjects[0] = new SimilarityCmpALowerB();   // optimum is min
	metric_names[1] = "mrsd"; metrics[1] = MRSD; cmpObjects[1] = new SimilarityCmpAGreaterB(); // optimum is max
	metric_names[2] = "ncc";  metrics[2] = NCC;  cmpObjects[2] = new SimilarityCmpALowerB();   // optimum is min
	metric_names[3] = "cch";  metrics[3] = CCH;  cmpObjects[3] = new SimilarityCmpAGreaterB(); // optimum is max
	metric_names[4] = "nmi";  metrics[4] = NMI;  cmpObjects[4] = new SimilarityCmpAGreaterB(); // optimum is max
	metric_names[5] = "msh";  metrics[5] = MSH;  cmpObjects[5] = new SimilarityCmpALowerB();   // optimum is min
	metric_names[6] = "mih";  metrics[6] = MIH;  cmpObjects[6] = new SimilarityCmpAGreaterB(); // optimum is max
	metric_names[7] = "emd";  metrics[7] = EMD;  cmpObjects[7] = new SimilarityCmpALowerB();   // optimum is min
	metric_names[8] = "gd";   metrics[8] = GD;   cmpObjects[8] = new SimilarityCmpAGreaterB(); // optimum is max

	// initialize pointer to null -> no metric included so far
	for (int i = 0; i < VOLE_NUMBER_OF_METRICS; i++)
		metrics_obj[i] = NULL;
}

void SimilarityMeasureCore::setConfig(SimilarityMeasureConfig *config) {
	this->config = config;
	for (unsigned int i = 0; i < config->metrics.size(); ++i)
		metrics_obj[static_cast<int>(config->metrics[i])] = getMetric(config->metrics[i]);
}

SimilarityMeasureCore::~SimilarityMeasureCore() {
	for (int i = 0; i < VOLE_NUMBER_OF_METRICS; ++i)
		delete metrics_obj[i];	// delete checks for 0 pointer itself!
}

SimilarityMeasure<unsigned char> *SimilarityMeasureCore::getMetric(Metric m) {
	switch(m) {
		case MS:   return new MeanSquares<unsigned char>();
		case MRSD: return new MeanReciprocalSquareDifference<unsigned char>();
		case NCC:  return new NormalizedCrossCorrelation<unsigned char>();
		case CCH:  return new CorrelationCoefficientHistogram<unsigned char>();
		case NMI:  return new NormalizedMutualInformation<unsigned char>();
		case MSH:  return new MeanSquaresHistogram<unsigned char>();
		case MIH:  return new MutualInformationHistogram<unsigned char>();
		case EMD:  return new EarthMoversDistance<unsigned char>();
		case GD:   return new GradientDifference<unsigned char>();
		default:   return NULL;
	};
}

// set matches[vole::Metric] to the worst possible value
void SimilarityMeasureCore::initMeasurements(std::vector<double> &matches) {
	if (matches.size() != VOLE_NUMBER_OF_METRICS) matches.assign(VOLE_NUMBER_OF_METRICS, 0.0);
	for (int i = 0; i < VOLE_NUMBER_OF_METRICS; ++i)
		matches[i] = cmpObjects[i]->extremalValue();
}

// set matches[vole::Metric] to the worst possible value, initialises all match points to (-1,-1)
void SimilarityMeasureCore::initMeasurements(std::vector<double> &matches, std::vector<cv::Point_<int> > &bestPositions) {
	initMeasurements(matches);
	bestPositions.assign(VOLE_NUMBER_OF_METRICS, cv::Point_<int>(-1, -1));
}

/// registriert einen Vergleichswert, wenn er eine Verbesserung beinhaltet (in matches[i], points[i])
void SimilarityMeasureCore::updateBestMetric(Metric m, cv::Point_<int> p, double value, std::vector<double> &matches, std::vector<cv::Point_<int> > &points) {
	int i = static_cast<int>(m);
	if (cmpObjects[i]->cmp(value, matches[i])) {
		matches[i] = value;
		points[i].x = p.x;
		points[i].y = p.y;
	}
}

void SimilarityMeasureCore::getSimilarity(std::vector<double> &distances) {
	if (distances.size() != config->metrics.size())
		distances.assign(config->metrics.size(), 0.0);
	for (unsigned int i = 0; i < config->metrics.size(); ++i) {
		distances[i] = metrics_obj[static_cast<int>(config->metrics[i])]->getSimilarity(img1, img2);
	}
}

void SimilarityMeasureCore::setImages(cv::Mat_<unsigned char> image1, cv::Mat_<unsigned char> image2)
{ img1 = image1; img2 = image2; }

std::string SimilarityMeasureCore::getMetricName(Metric m) {
	return std::string(metric_names[m]);
}

std::string SimilarityMeasureCore::metricToString(Metric m)
{ return metric_names[static_cast<int>(m)]; }

int SimilarityMeasureCore::metricToIndex(Metric m) { return static_cast<int>(m); }

Metric SimilarityMeasureCore::stringToMetric(std::string m) {
	
	for (int i = 0; i < VOLE_NUMBER_OF_METRICS; ++i) {
		if (m.compare(metric_names[i]) == 0) return metrics[i];
	}
	return NO_METRIC;
}

bool SimilarityMeasureCore::parseSelectedMetrics(std::string selectedMetrics, std::vector<Metric> &metrics) {
	metrics.clear();
	if (selectedMetrics.length() < 1) {
		std::cerr << "\"metrics\" is empty. At least one metric must be given, aborted." << std::endl;
		return false;
	}
	std::replace (selectedMetrics.begin(), selectedMetrics.end(), ',', ' '); 
	std::string tmp;
	std::stringstream s;
	s << selectedMetrics;
	while (s >> tmp) {
		Metric t = stringToMetric(tmp);
		if (t == NO_METRIC) {
			std::cerr << "\"" << tmp << "\" is not a known metric, skipped. Metrics must be from {ms,mrsd,ncc,cch,nmi,msh,mih,emd,gd}" << std::endl;
			continue;
		}
		metrics.push_back(t);
	}
	if (metrics.size() == 0) {
		std::cerr << "No known metric found, aborted." << std::endl;
		return false;
	} else {
		return true;
	}
}



}

