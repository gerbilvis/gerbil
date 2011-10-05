#ifndef SOM_TRAINER_H
#define SOM_TRAINER_H

#define FULL_VERBOSITY false

#include "edge_detection_config.h"
#include "neuron.h"
#include "self_organizing_map.h"
#include <similarity_measure.h>
#include <multi_img.h>
#include <cv.h>
#include <cmath>
#include <limits.h>

class SOMTrainer {

public:

	/**
	* Simple SomTrainer constructor
	*
	* @param	map pointer to SOM which should be used
	* @param	image pointer to a multi_img
	* @param	name filename of the multispectral image
	*/
	SOMTrainer(SOM &map, const multi_img &image,
	           const EdgeDetectionConfig &conf);

	//! Destructor
	~SOMTrainer() {}

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
	void feedSample(const multi_img::Pixel &input);

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
	cv::Mat1d umatrix(bool write =true);

protected:
	vole::SimilarityMeasure<multi_img::Value> *distfun;

	SOM &som;
	const multi_img &input;
	const EdgeDetectionConfig &config;

	cv::Mat3f m_img_som;
	cv::Mat3f m_3d_som;
	cv::Mat3f m_img_msi;
	cv::Mat1f m_img_bwsom;
	cv::Mat_<unsigned int> m_rank;
	
	cv::Mat1d *m_edgeWeights;
	cv::Mat1d umap;
	cv::Mat1d m_bmuMap;

	int m_nr;
	int m_foundEdges;
	int maxIter;
	int currIter;

	bool m_withGraph;
	bool m_withUMap;
};

#endif // SOM_TRAINER_H
