/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>,
	Felix Lugauer, Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_SIMILARITY_MEASURE_CORE_H_
#define VOLE_SIMILARITY_MEASURE_CORE_H_

#include "similarity_measure.h"
#include "similarity_measure_config.h"

#include <vector>


#ifndef VOLE_NUMBER_OF_METRICS
#define VOLE_NUMBER_OF_METRICS 9
#endif

namespace vole {

class SimilarityCmp {
	public:
	virtual bool cmp(const double &a, const double &b) { return false; }
	virtual double extremalValue() { return 0.0; }
};
// minimize
class SimilarityCmpALowerB : public SimilarityCmp {
	public:
		virtual bool cmp(const double &a, const double &b) { return a < b; }
		virtual double extremalValue() { return std::numeric_limits<double>::infinity(); }
};
// maximize
class SimilarityCmpAGreaterB : public SimilarityCmp {
	public:
		virtual bool cmp(const double &a, const double &b) { return a > b; }
		virtual double extremalValue() { return -std::numeric_limits<double>::infinity(); }
};

class SimilarityMeasureCore {
	public:
		SimilarityMeasureCore();
		SimilarityMeasureCore(SimilarityMeasureConfig *config);
		~SimilarityMeasureCore();

		void setConfig(SimilarityMeasureConfig *config);
		
		// eher ROI position... setzt patchPosX, patchPosY, patchWidth, patchHeight
//		void setTemplatePosition(int x, int y, int width, int height);

		// ordnet jeder Metric eine Nummer zu
		Metric stringToMetric(std::string m);
		std::string metricToString(Metric m);
		int metricToIndex(Metric m);

		SimilarityMeasure<unsigned char> *getMetric(Metric m);
		void getSimilarity(std::vector<double> &distances);

		void setImages(cv::Mat_<unsigned char> image1, cv::Mat_<unsigned char> image2);

		bool parseSelectedMetrics(std::string selectedMetrics, std::vector<Metric> &metrics);

		/// registriert einen Vergleichswert, wenn er eine Verbesserung beinhaltet (in matches[i], points[i])
		void updateBestMetric(Metric m, cv::Point_<int> p, double value, std::vector<double> &matches, std::vector<cv::Point_<int> > &points);

		/// set matches[vole::Metric] to the worst possible value
		void initMeasurements(std::vector<double> &matches);
		/// set matches[vole::Metric] to the worst possible value, initialises all match points to (-1,-1)
		void initMeasurements(std::vector<double> &matches, std::vector<cv::Point_<int> > &bestPositions);

	protected:
		// assigns Metric to numbers, optimization criteria etc.
		void initClass();

		// ordnet jeder Metric eine Nummer zu und befuellt metrics[i] mit einer neuen Instanz der Metric
		int setMetricID(std::string m);

		// string darstellung der Metric
		std::string getMetricName(Metric m);
		
		SimilarityMeasureConfig *config;

		cv::Mat_<unsigned char> img1;
		cv::Mat_<unsigned char> img2;
		// patch features
		int patchPosX;
		int patchPosY;
		int patchWidth;
		int patchHeight;
		std::string output;
		
		SimilarityCmp *cmpObjects[VOLE_NUMBER_OF_METRICS];
		SimilarityMeasure<unsigned char> *metrics_obj[VOLE_NUMBER_OF_METRICS];
		std::string metric_names[VOLE_NUMBER_OF_METRICS];
		Metric metrics[VOLE_NUMBER_OF_METRICS];
};


}
#endif // VOLE_SIMILARITY_MEASURE_CORE_H_
