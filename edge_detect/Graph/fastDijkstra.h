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


#ifndef FAST_DIJKSTRA_H
#define FAST_DIJKSTRA_H
 
#include <iostream>
#include <list>
#include <vector>
#include <algorithm>
#include <cassert>
#include <climits>
#include "cv.h"

/**
*
*	Class of nodes inside the adjacence list
*	The node defines the 'end node' of the edge, weighted by 'fwt;
*	The 'start node' is implicitly defined by the position in the adjacence list adj
*
*/
class Node
{
public:

	//! Constructor
	Node(unsigned int iVNo, double fWt)
	{
		iVertexNo = iVNo;
		distance = fWt;
		edgeWeight = fWt;
		weightedDistance = fWt;
		
	}
	//!	Default destructor
	~Node() {}
	
	unsigned int iVertexNo;
	double distance;					//! \brief physical distance to neighbors, i.e. 1 or sqrt(2) for diagnoal edges, respectively
	double weightedDistance;	//! \brief scaled weight
	double edgeWeight;				//! \brief new edge weight according to dissimilarity

};

//! Implementation of Dijkstra's shortes path algorithm based on adjacence lists
class fastDijkstra
{
	public:
		
		//! Constructs a graph with 'nodes' nodes
		fastDijkstra(unsigned int nodes);
		
		//! Display contents of pie and dist (debug purpose only)
		void dumpArrays();
		
		//! Insert a new (weighted) edge
    void addEdge(unsigned int from,unsigned int to, double edgeWeight);
		
		//! Remove a (weighted) edge
		void removeEdge(unsigned int indexA, unsigned int indexB);
		
		//! Return edge weight between 'from' and 'to'
		double getEdgeWeight(unsigned int from,unsigned int to);
		
		//! Set edge weight between 'from' and 'to'
		void setEdgeWeight(unsigned int from,unsigned int to, double weight);

		//! Scaling function for weighted edges (corresponds to weighting functiion omega() in the written thesis
		void scaleDistances(double scaling);
		
		/**
		*	Returns shortest path between two nodes
		*
		*	@param	source			source node
		*	@param	destination	destination node
		*	@param	weighted		if true, 'weigthedDistance' is used for the computation, otherwise 'distance' is used.
		*
		*/
    double shortestPath(unsigned int source, unsigned int destination, bool weighted=false);

		/**
		*	Returns shortest path between two nodes, but computes additionally the distances among all nodes occuring in the graph
		*
		*	@param	source			source node
		*	@param	destination	destination node
		*	@param	weighted		if true, 'weigthedDistance' is used for the computation, otherwise 'distance' is used.
		*
		*/
		double allPath(unsigned int source, unsigned int destination, bool weighted=false);
		
		/**
		*	Prints the intermediate nodes on the path between two nodes on console
		*
		*	@param	s	source node
		*	@param	d	destination node
		*
		*/		
		void printPath(unsigned int s, unsigned int d);

		/**
		*	Stores the intermediate nodes on the path between two nodes in a vector
		*
		*	@param	s	source node
		*	@param	d	destination node
		*
		*/				
		void path(unsigned int src, unsigned int dest, std::vector<unsigned int> *v);

		//! Returns number of edges
    unsigned int getEdgeSize(){return adj->size();}
    
		/**
		*	Returns the (cached) shortest path between two nodes. The display of the intermediate nodes is optional.
		*
		*	@param	indexA								Index of source node
		*	@param	indexB								Index of destination node
		*	@param	getWeightedDistance		if true, 'weigthedDistance' is used for the computation, otherwise 'distance' is used.
		*	@param	showPath							The distance is computed explicitly (without caching) and the intermediate traversed nodes are displayed on console
		*
		*/
    double getDistance(unsigned int indexA, unsigned int indexB,bool getWeightedDistance, bool showPath);
		
		//! Returns a vector containing the traversed node indices
		std::vector<unsigned int> getPath(unsigned int indexA, unsigned int indexB);
		
		//! Returns a vector containing the traversed node indices when using weighted edges
		std::vector<unsigned int> getWeightedPath(unsigned int indexA, unsigned int indexB);
		
		//! Display content of distance cache (debugging purpose only)
		void showCache();
		
		//! Returns the diameter of the graph
		double getDiameter();
		
		/**
		*	\brief Return characteristic path length
		*	
		*	The charachteristic path length describes the median of the mean distance for all nodes to all other nodes
		*/
		double characteristicPathLength();
		
		//! Scaling of weight. 0.0 <= scale
		double scaleWeight(double weight, double scale);
		
		//! Default destructor
		~fastDijkstra();
	
	private:
		unsigned int m_nodes_number;
		unsigned int m_width;
		std::list<Node*> *adj;
		
	unsigned int *colour;	//[m_nodes_number];
	double *dist;					//[m_nodes_number];
	unsigned int *pie;		//[m_nodes_number];

	double m_weightScaling;

	cv::Mat2d m_distanceCache;
		
};

#endif