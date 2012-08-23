#ifndef VOLE_SIMILARITY_MEASURE_CONFIG_H
#define VOLE_SIMILARITY_MEASURE_CONFIG_H


#include "vole_config.h"

#include "cv.h"

#include <iostream>
#include <string>
#include <vector>

namespace vole {

	enum Metric {
		NO_METRIC = -1,
		MS = 0,
		MRSD = 1,
		NCC = 2,
		CCH = 3,
		NMI = 4,
		MSH = 5,
		MIH = 6,
		EMD = 7,
		GD = 8
	};

	class SimilarityMeasureConfig : public Config {
		public:
		SimilarityMeasureConfig(const std::string &prefix = std::string());

		// graphical output on runtime?
		bool isGraphical;
		// input data
		std::string input_file1;
		// input data2
		std::string input_file2;
		// working directory
		std::string output_directory;
		// subwindow coordinates in image 1, ","-separated
		std::string win1;
		// subwindow coordinates in image 2, ","-separated
		std::string win2;
		// ','-separated list of metrics to apply on the images
		std::string selected_metrics;

		/// parsed subwindow coordinates for image 1
		cv::Rect_<int> window1;
		/// parsed subwindow coordinates for image 2
		cv::Rect_<int> window2;
		/// parsed list of metrics
		std::vector<Metric> metrics;

		virtual std::string getString() const;

		#ifdef VOLE_GUI
			virtual QWidget *getConfigWidget();
			virtual void updateValuesFromWidget();
		#endif// VOLE_GUI

		protected:

		#ifdef WITH_BOOST
			virtual void initBoostOptions();
		#endif // WITH_BOOST

		#ifdef VOLE_GUI
		// qt data structures 
		#endif // VOLE_GUI
	};

}

#endif // VOLE_SIMILARITY_MEASURE_CONFIG_H
