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


#ifndef SELF_ORGANIZING_MAP_H
#define SELF_ORGANIZING_MAP_H

#include <vector>	//std library
#include "neuron.h"
#include "multi_img.h"

#include "edge_detection_config.h"
//graph includes

#include "Graph/smallWorldGraph.h"
#include "Graph/completingGraph.h"
#include "Graph/graph.h"
#include "Graph/misc.h"
#include "Graph/solution.h"


class SelfOrganizingMap {

public:

	/**
	* SelfOrganizingMap constructor
	*
	* @param	width Width of the SOM grid
	* @param	height Height of the SOM grid
	* @param	dimension Size of each neuron at each grid
	*/
	SelfOrganizingMap(const EdgeDetectionConfig *conf, int dimension)
	: m_vector_dim(dimension), m_width(conf->som_width), m_height(conf->som_height),
	  m_graph(conf->graph_withGraph),m_umap(conf->withUMap) , m_conf(conf)
	{
		init();
		randomizeNeurons(0., 1.);
	}

	/**
	* SelfOrganizingMap loads and restores som data from a given file,
	* which has been created previously.
	* Further information about the file format is given at the
	* documentation about the load() / save() function of the SOM.
	*
	* @param	somFile Filename of a som data file to restore som values
	*/
	SelfOrganizingMap(std::string somFile) { load(somFile); }

	/**
	* Destructor of SelfOrganizingMap deleting the memory created on heap
	*
	*/
	~SelfOrganizingMap() {
		for(int y = 0; y < m_height; y++) {
			for(int x = 0; x < m_width; x++ ) {
				delete m_neurons[y][x];
			}
			delete[] m_neurons[y];
		}

		delete m_neurons;
	}

	/**
	* Creates SOM data structure - the grid of neurons - at the heap memory
	* and sets each value of a neuron to zero.
	*/
	void init();
    
    /**
    * Creates CompletingGraph structure
    */
    void initCompletingGraph(msi::Configuration& conf);

    
    /**
    * Calculates the weight between two nodes
    * Calls double CompletingGraph::calculateEdgeWeight(unsigned int node1, unsigned int node2, int mode=0);
    * @param node1 first node
    * @param node1 second node
    * @param mode 0: Euclidean Distance 1: Spectral Angle Similarity 2: Not implemented yet
    * @return the edge weight or DBL_MAX if no edge exists
    */
//     double calculateEdgeWeight(unsigned int node1, unsigned int node2, int mode=0);

	/**
	* Reallocates the size of each neuron
	*/
	void reallocate(unsigned int bands);

	/**
	* Uniformly randomizes each neuron within the given interval.
	*
	* @param	dataLow Upper bound of uniform range
	* @param	dataHigh lower bound if uniform range
	*/
	void randomizeNeurons(double dataLow, double dataHigh);

	/**
	* Finds the neuron in the SOM grid which has the closest distance
	* to the given input vector and returns its position in the grid.
	*
	* @param	input Neuron to which closest neuron in SOM will be determined
	* @return	Position of the neuron in x,y coordinates
	*/
	cv::Point identifyWinnerNeuron(const multi_img::Pixel &input);

    
    /**
    * Generates a lookup table of size "size" for the neurons in the SOM grid for faster access
    * @param  msi Pointer to the multispectral image
    */    
    void generateLookupTable(const multi_img &msi);
    
    /**
    * Returns the position of the closest neuron by searching the lookup table 
    * @param  x x coordinate to look for
    * @param  y y coordinate to look for
    * @return Position of the closest neuron in x,y coordinates
    */
    cv::Point identifyWinnerNeuron(int y, int x);

	/**
	* Returns a pointer to the neuron at the given grid position
	*
	* @param	x x-coordinate
	* @param	y y-coordinate
	* @return	Pointer to neuron
	*/
	inline Neuron* getNeuron(int x, int y) { return m_neurons[y][x]; }

	/**
	* Saves all SOM data into a specified file.
	*
	* File format:
	* line 1: height width dimension
	* line 2-n: N_i_0 N_i_1 ... N_i_m
	* where N_i is the i-th of n neurons in the SOM.
	* Each neuron contains of m double values.
	*
	* @param	output Filename to store SOM values
	* @return	Error variable
	*/
	int save(std::string output, std::string params);

	/**
	* Restores SOM data from a given SOM file.
	*
	* @param	input Filename of a SOM data file.
	* @return	Error variable
	*/
	std::string load(std::string input);

	inline std::string getParams() { return m_params; }


	//! Returns the width of the SOM grid
	inline int getWidth() { return m_width; }

	//! Returns the height of the SOM grid
	inline int getHeight() { return m_height; }


	//! Returns dimensionality of the SOM ( equal to neuron dimensionality)
	inline unsigned int getDimension() { return m_vector_dim; }


  //!	Returns pointer to msi::CompletingGraph
	inline msi::Mesh* getGraph(){return msi_graph;}
		

  //! Returns true if graph topology is used
  inline bool withGraph(){return m_graph;}		


protected:
	Neuron*** m_neurons;				///< Pointer structure representing the SOM grid
	cv::Point** m_lookupTable; ///< pre calculated positions of closest neurons

	int m_vector_dim;		///< Dimension of each neuron / the SOM
	int m_width;		  	///< Width of SOM grid
	int m_height;		  	///< Height of SOM grid
	bool m_graph;       ///< If graph is used
	bool m_umap;        ///< If weight map is used

	std::string m_params;	///< Parameter of the SOM when being loaded from file

	cv::Mat1d m_edgeWeights;
	msi::Mesh *msi_graph;
	const EdgeDetectionConfig *m_conf;
};


#endif // SELF_ORGANIZING_MAP_H
