#ifndef SOM_TRAINER_H
#define SOM_TRAINER_H

#include "edge_detection_config.h"
#include "neuron.h"
#include "som.h"

#include <multi_img.h>

#include <cmath>
#include <limits.h>

// forward declarations
namespace vole {
class ProgressObserver;
}

class SOMTrainer {

public:

	SOMTrainer(SOM *map, const multi_img &image,
							 const vole::EdgeDetectionConfig &conf);

	SOMTrainer(SOM *map, const multi_img &image,
			   const vole::EdgeDetectionConfig &conf,
			   volatile bool& abort,
			   vole::ProgressObserver *po = NULL);

	//! Destructor
	~SOMTrainer() {}
	static SOM *train(const vole::EdgeDetectionConfig &conf,
					  const multi_img& img);

	static SOM *train(const vole::EdgeDetectionConfig &conf,
					  const multi_img& img,
					  volatile bool& abort,
					  vole::ProgressObserver *po);

	/**
	* Controls the training process using random vectors of the multispectral image
	* and the SOM network. 'feedSample()' is called for each iteration.
	*/
	void feedNetwork();

	/**
	* Executes one iteration: Present input vector to the SOM
	* and find best matching neuron.'updateNeighborhood' is then
	* responsible for the adaption of the network.
	*/
	int feedSample(const multi_img::Pixel &input);

	/**
	* Calculates vector norm of 2 vectors
	*
	* @param  &p1 point 1
	* @param  &p2 point 2
	* @param  mode 0  manhattan RGB 1 euclidean RBG 2 angle RBG 3 Mahalonobis Distance
	* @return distance of type float
	*/
	float vectorDistance(const cv::Point &p1, const cv::Point &p2, int mode); // TODO: use sim_meas enum

	float vectorDistance(const cv::Point2f &p1, const cv::Point2f &p2);	// TODO: double method fuckup

	/**
	* Calculates shortest path/distance between two nodes in a graph.
	* The function distincts between weighted graphs and non-weighted graph, and returns
	*	the vector distance between two nodes, if the 'FDD' (force direct distance) option is enabled
	*
	* @param p1 index of point 1
	* @param p2 index of point 2
	* @return Shortest path/distance between two nodes
	*/
	double graphDistance(cv::Point2d &p1, cv::Point2d &p2);

	//!Calculates the wrap-around distance in the periodic SOM
	double wrapAroundDistance(cv::Point2d &p1, cv::Point2d &p2);

	/**
	* Generates a difference to neighbors SOM and reduces dimensionality
	* through euclidean distance to a scalar
	* that represents homogeneity of the SOM network.
	*/
	double generateBWSom();

	/**
	*	Calculates edge weights based on spectral dissimilarity (Euclidean distance)
	*
	* @param write write umap to disk
	*/
//	cv::Mat1d umatrix(bool write = true);

protected:
	SOM *const som;
	const multi_img &input;
	const vole::EdgeDetectionConfig &config;

	cv::Mat1f m_img_bwsom;
	cv::Mat1d umap;
	cv::Mat1d m_bmuMap; // warum eine double matrix?

	int currIter;
	volatile bool* abort;
	vole::ProgressObserver *po;
};

#endif // SOM_TRAINER_H
