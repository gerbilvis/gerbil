//	-------------------------------------------------------------------------------------------------------------	 	//
// 														Variants of Self-Organizing Maps																											//
// Studienarbeit in Computer Vision at the Chair of Patter Recognition Friedrich-Alexander Universitaet Erlangen		//
// Start:	15.11.2010																																																//
// End	:	16.05.2011																																																//
// 																																																									//
// Ralph Muessig																																																		//
// ralph.muessig@e-technik.stud.uni-erlangen.de																																			//
// Informations- und Kommunikationstechnik																																					//
//	---------------------------------------------------------------------------------------------------------------	//


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


// Parts of this code are based on implementations of Felix Lugauer

class SomTrainer {

public:

	/**
	* Simple SomTrainer constructor
	*
	* @param	map pointer to SOM which should be used
	* @param	image pointer to a multi_img
	* @param	name filename of the multispectral image
	*/
	SomTrainer(SelfOrganizingMap &map, const multi_img &image,
			   const EdgeDetectionConfig &conf, const std::string &name = "msi");

	//! Destructor
	~SomTrainer() {}

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

	//! Update a single neuron
	void updateSingle(const cv::Point &pos, const multi_img::Pixel &input, double weight);

	/**
	* Updates a spheric neigborhood around the winning neuron
	* using the distance between current neuron und current pixel vector.
	* Involves a time- and distance dependent weighting function.
	*/
	void updateNeighborhood(const cv::Point &pos, const multi_img::Pixel &input);
  
  
	/**
	* Updates a spheric neigborhood around the winning neuron
	* using the distance between current neuron und current pixel vector.
	* Involves a time- and distance dependent weighting function.
	*/
	void updateGraphNeighborhood(cv::Point &pos, const multi_img::Pixel &input);

	//! Offline training for Self-Organizing Map
	double offlineTraining(int leaveOut = 0);

	/**
	* Computes the current learning factor and radius for a specific iteration.
	* Time variant and involving a gaussian distribution weighted neighborhood.
	*/
	void computeAdjustedWeights(double &learn, double &radius);

	/**
	* Generates the 2D grayscale representation of the multispectral image
	* using the 1D SOM network and the multispectral image by finding
	* the best matching neuron to each pixel vector of the multispectral image.
	*
	* @return Rank image which is the result of the dimensionality reduction
	*/
	cv::Mat generateRankImage();

	/**
	* Generates the 2D grayscale representation of the multispectral image
	* using the 2D SOM network and the multispectral image by finding
	* the best matching neuron to each pixel vector of the multispectral image.
	* Additionally involves linearization of the two dimensional SOM
	* using an appropriate hilbert or peano curve.
	*/
	cv::Mat generateRankImage(cv::Mat_<unsigned int> &rank);

	/**
	* Applies the canny edge detector to the rank/grayscale image.
	* Hysteresis parameter control the edge detection result.
	*
	* @param	h1 lower hysteresis threshold parameter
	* @param	h2 upper hysteresis threshold parameter
	* @return	Edge image of type cv::Mat
	*/
	cv::Mat generateEdgeImage( double h1, double h2);
	
	/**
	*
	* Compute directional distance images using a fake Sobel operator
	*
	*	@param dx	Horizontal distance map
	*	@param dx	Vertical distance map
	*	@param mode	0: Sobel fake, 1: Scharr fake
	*
	*/
  void getEdge( cv::Mat1d &dx, cv::Mat1d &dy, int mode);
    
	//! Computes Euclidean distance between two points
	double vectorDistance(const cv::Point2d &p1, const cv::Point2d &p2);
	
	/**Calculates intersections on the SOM borders 'border' (2D matrix that characterizes paramterform of line), 
	*	with the line defined by p1 and p2, writes the intersection in intersection and returns true on success.
	* Needed for periodic SOMs to calculate shortest wrap-around distance
	*/
	bool findBorderIntersection(cv::Point2d &p1, cv::Point2d &p2, cv::Point2d &intersection,cv::Mat1d &border);

  /**
  * Calculates shortest path/distance between two nodes in a graph.
	* The function distincts between weighted graphs and non-weighted graph, and returns
	*	the vector distance between two nodes, if the 'FDD' (force direct distance) option is enabled
	*
  * @param p1 index of point 1
  * @param p2 index of point 2
  * @return Shortest path/distance between two nodes
  */
  double graphDistance( cv::Point2d &p1, cv::Point2d &p2);	
	
	//!Calculates the wrap-around distance in the periodic SOM
	double wrapAroundDistance( cv::Point2d &p1, cv::Point2d &p2);

	/**
	* Generates a difference to neighbors SOM and reduces dimensionality
	* through euclidean distance to a scalar
	* that represents homogeneity of the SOM network.
	*/
	double generateBWSom();

	/**
	* RGB visualization of SOM network
	* using CIE X Y Z and D65 illumination.
	*
	* @param write write RGB-Image to disk
	*/
	void displaySom(bool write = true);
	
	//! Write png of SOM to disk if current iteration is equal to iter 
	void writeSom(int iter);
	
	//! Computes Euclidean distance of two 2-dimensional points
	inline double euclideanDistance(cv::Point2d p1,cv::Point2d p2)
	{
		return std::sqrt( ((p1.x - p2.x)*(p1.x - p2.x)) + ((p1.y - p2.y) * (p1.y - p2.y) ));
	}
	
	//! Computes Manhattan distance of two 2-dimensional points
	inline double manhattanDistance(cv::Point2d p1,cv::Point2d p2)
	{
		return (std::abs(p1.x - p2.x) + std::abs(p1.y - p2.y));
	}	

	/**
	* RGB visualization of multispectral image
	* using CIE X Y Z and D65 illumination. (deprecated)
	*
	* @param write write RGB-Image to disk
	*/
	void displayMSI(bool write = true);
  
  /**
  * Node representation of graph in dot file
  *
  * @param write write converted cv::Mat to disk
  */
  void displayGraph(bool write = true);  
	
  /**
  * Counts number of each neuron being the BMU (winning neuron)
  *
  * @param write write grayscale coded image to disk
  */
  void displayBmuMap(bool write = true); 	
  
  /**
  * Distances between SOM nodes
  *
  * @param write write image to disk
  */
  void displayDistanceMap(bool write = true);
	
	
	/**
	*	Calculates edge weights based on spectral dissimilarity (Euclidean distance) 
	*	
	* @param write write umap to disk
	*/
	cv::Mat1d umatrix(bool write =true);


	/**
	* Multispectral analysis tool that visualizes both the multispectral
	* image and SOM data as well as the resulting rank image.
	* Very useful to examine the conjunction between the SOM learning of the
	* multispectral input and the mapping to the rank image.
	*/
	void compareMultispectralData();
	
	/**
	*	\brief Visualize distances between two nodes
	*
	*	Left-click 	: Select first node
	*	Right-click 	: Select second node
	* If two nodes are selected, the computed shortest paths are visualized graphically, and the distances
	*	are displayed on console output
	*
	*/
	void displayGraphDistances();

	//!	Deprecated
	void processStatistics();

	/**
	* Helper function to generate specific output names
	* consisting of the current parameter set.
	*/
	std::string getFilenameExtension();

	//!	Deprecated version of getFilenameExtension()
	std::string getParams(bool measurements = false);

	std::vector<double> getStatistics();
	
	//! Collects current graph settings in a string
	std::string graphProperties(){return m_graphProperties.str();};


protected:
	vole::SimilarityMeasure<multi_img::Value> *distfun;

	SelfOrganizingMap &m_som;
	const multi_img &m_msi;
	const EdgeDetectionConfig &config;

	cv::Mat1f m_img_rank;
	cv::Mat m_edge;
	cv::Mat3f m_img_som;
	cv::Mat3f m_3d_som;
	cv::Mat3f m_img_msi;
	cv::Mat1f m_img_bwsom;
	cv::Mat_<unsigned int> m_rank;
	
	cv::Mat1d *m_edgeWeights;
// 	cv::Mat1d m_distanceMap;
	cv::Mat1d umap;
	cv::Mat1d m_bmuMap;

	std::string m_msiName;

	double m_learningStart;
	double m_learningEnd;
	double m_radiusStart;
	double m_radiusEnd;
	double m_sw_radiusStart;

	double m_evolution;
	double m_quantizationError;
	double m_errorEntropy;
	double m_balance;

	int m_nr;
	int m_foundEdges;
	int m_iter;
	int m_currIter;
	int m_widthSom;
	int m_heightSom;
	int m_bands;
	int m_verbosity;
  
	bool m_withGraph;
	bool m_withUMap;
	
	std::stringstream m_graphProperties;
  
  
private:
    
	msi::Mesh *msi_graph;
  
};

#endif // SOM_TRAINER_H
