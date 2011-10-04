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

// Parts of this code are based on implementations of Johannes Michael Jordan

#ifndef MSI_COMPLETINGGRAPH_H
#define MSI_COMPLETINGGRAPH_H

#include <utility>
#include <limits.h>

#include "graph.h"
#include "misc.h"

#include "multi_img.h"

#include "dijkstra.h"
#include "fastDijkstra.h"

#include "cv.h"
#include "highgui.h"


namespace msi{
		
	class CompletingGraph : public Graph{
		/** relational operator for Node degrees */
		struct mingrade
		{
			bool operator()(const Node* s1, const Node* s2) const
			{
				return s1->getDegree() < s2->getDegree();
				// hier auf keinen Fall mischen!
			}
		};

		typedef std::multiset<const Node*>::iterator nbgIter;

		public:

      CompletingGraph(const Configuration &c,unsigned int n, bool directed);
			/** Overwrites Graph::remove to also remove the Node from nodesByGrade. */
			virtual void remove(unsigned int index);
			
			virtual void nextIter();

			virtual unsigned int addMember(Neuron *n);
			
			Neuron * deleteMember(unsigned int index);
      
			//! Set pointer to multi_img
      inline void setMSI(const multi_img *msi){m_msi=msi;}
      
      inline unsigned int getMinimumEdgeCount(){return minimumEdgeCount;}
			
			//! Give the size of the reserved space for Nodes
      unsigned int getSize(){return Graph::getSize();}
      
      //! Get node with index idx
      const Node* getNode(unsigned int idx){return Graph::getNode(idx);};

			//! Get the edges of node idx
			const std::vector<Node*>& getEdges(unsigned int idx){return Graph::getNode(idx)->getEdges();}

      //! True if topography is periodic 
      bool periodic(){return m_isPeriodic;}

      /**
      * Remove an edge anywhere inside the graph. 
      * @return wether an edge could be removed (or the graph is already at minimum size)
      **/
      virtual bool removeRandomEdge();

		protected:
			/**
			 * Add a new edge anywhere inside the graph. Used by the completer
		 	 * @return wether an edge could be added (or the graph is already complete)
			 **/
			virtual bool addRandomEdge();
     
      /**
      * randomizes nodesByGrade
      * @param field array with nodes
      * 
      */
      void randomizeNodes(unsigned int *field );
      
			//! Remove edge connecting 'from' and 'to'
      bool removeEdge(unsigned int from, unsigned int to);
      
      /**
      * Rewire graph such that it fulfills small world graph conditions :
      * high clustering and short average distance
      * @param p parameter that controls randomness of graph:
      * p == 0.0 : totally regular p == 1.0 totally random
      * @param type create graph according BETA or PHI model
      */
      virtual void createSmallWorld(double p, sw_type type);

      /**
      * Check if two nodes share a common "friend"
      *  @param u first node
      *  @param v second node 
      *  @param candidate index of found candidate or UINT_MAX if no shared "friend" exists
      */
      bool checkMutualfriends(const Node *u, const Node *v, unsigned int &candidate);
   
      /**
      * Calculates the weight between two nodes
      * @param node1 first node
      * @param node1 second node
      * @param mode 0: Euclidean Distance others: Not implemented yet
      * @return the edge weight or FLT_MAX if no edge exists
      */
      double calculateEdgeWeight(unsigned int index1, unsigned int index2, int mode);
      

			//! Holds the graph configuration
			Configuration conf;
			
			/** Tells wether nodes where added or removed.
			*	This disturbs the completion process; random edge insertion methods could
			*	need to fallback to the general method if this happens. 
			*	(Not used in small world graphs)
			*
			*/
      bool intruder;
            
			//! Pointer to multi_img
      const multi_img *m_msi;
      
      ///important only for small world graphs
      unsigned int minimumEdgeCount;
      
			//! Number of maximum iterations
      unsigned int maxIter;
      
      //! True, graph has periodic structure
      bool m_isPeriodic;
				

            
		private:
			/** Initialize the completer. */
			void initCompleter();
			/** Proceed completing. Does the work necessery in one iteration. */
			void proceedCompleter();
			/** Re-init the completer (after nodes were added/removed) */
			void restartCompleter();
      
			/** Node multiset sorted by grade, used for completion */
			std::multiset<const Node*, mingrade> nodesByGrade;
			/** actual start iteration for completion */
			unsigned int start;
			/** actual end iteration for completion */
			unsigned int end;
			/** completer: new edges per iteration */
			double edgesPerIter;
			/** completer: absolute number of edges needed in this iteration */
			double edgesWantedNow;
	};
  
  class Mesh : public CompletingGraph {

    public:
			/**
			* 2-dimensional substrate
			*	Depending on the configuration c the topology may be periodical or become
			*	a Small-World graph
			*
			*	@param c Holds the graph configuration
			*	@param n the number of reserved nodes
			*
			*/
			Mesh(const Configuration & c, unsigned int n);
			
			//! Switch to the next iteration
      void nextIter();

			//! Allows to dynamically extend the graph (not used in this implementation)
      unsigned int addMember(Neuron *n);
      
	   	/**
			* Vary the influence of the weighting function.
			*	Corresponds to 'alpha' in the written thesis that control the weighting function omega
			*	scale = 0.0 : Edge weights follow the graph topology
			*	scale = 1.0 : Edge weights only depend on the dissimilarity measure
			*/
			void scaleDistances(double scale = 1.0);

			//! Setup the Shortest Path computation
			void initDijkstra();
			
			//! Compute full n x n distance matrix, where n corresponds to the number of nodes
			void precomputePaths(bool weighted);

      //! print the graph to dot file
      void print(std::ostream& o, const std::string& n) const;
			
      //! Test function to check correct graph setup
      void checkNeighbors();
			
			//! Return clustering coefficient
			double graphClustering();

			//! Return diameter (= longest shortest path between two nodes occuring in the graph)
			double getDiameter(){return m_dijkstra->getDiameter();}
			
			/**
			*	\brief Return characteristic path length
			*	
			*	The charachteristic path length describes the median of the mean distance for all nodes to all other nodes
			*/
			double getcharacteristicPathLength(){return m_dijkstra->characteristicPathLength();}
			
			//! Set the edge weight for the edge connecting 'from' and 'to'
			void setEdgeWeight(unsigned int from,unsigned int to, double weight){m_dijkstra->setEdgeWeight(from,to,weight);}
			
			//! Returns a vector with the traversed path between 'index1' and 'index2'
			std::vector<unsigned int> getPath(unsigned int index1, unsigned int index2){return m_dijkstra->getPath(index1,index2);}
			//! Returns a vector with the traversed path between 'index1' and 'index2' when weighted edges are used
			std::vector<unsigned int> getWeightedPath(unsigned int index1, unsigned int index2){return m_dijkstra->getWeightedPath(index1,index2);}
			
			/*! Calculates distance between node index1 and index2
			* @param weighted	if true, return weighted distance
			*/
			double getDistance(unsigned int index1,unsigned int index2, bool weighted, bool showPath=false);

			/** 
			*	\brief Update function for a neuron/node
			*
			*	@param	origin		pointer to center node of neighborhood function (BMU)
			*	@param	depthMax	radius of neighborhood function
			*	@param	learning	current learning rate
			*	@param	input			multi_im pixel that is used to update
			*/
			unsigned int updateNeighborhood( const msi::Node *origin, double depthMax ,double learning, const multi_img::Pixel &input);

    protected:
      /**
       * helper function which gives the next possible neighbors of a node in the lattice,
       * taking into account empty slots
       * @param index start index for the search
       * @param dir direction in the lattice (0 left, 1 right, 2 top, 3 down)
       * @return index of next neighbor found
       **/
      unsigned int findNextNeighbor(unsigned int index, unsigned int dir);
						
			
      /// width of the lattice (sqrt(n) at start)
      unsigned int width;
      /// height of the lattice (sqrt(n) at start)
      unsigned int height;			
      /** 
			*	number of edges of default t x t 2D Mesh
      * according the rule: #Edges = 2N - 2t; N = number nodes, t = sqrt(N)
      */
      unsigned int noEdges;
      
      ///additional random edges
      unsigned int noRandEdges;
			
		protected: 
			
			fastDijkstra *m_dijkstra;
  };
	
}//end of namespace msi

#endif
