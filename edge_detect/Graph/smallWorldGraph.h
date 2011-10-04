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


#ifndef SMALLWORLDGRAPH_H
#define SMALLWORLDGRAPH_H

#include "Graph/graph.h" 
#include "Graph/misc.h" 
#include "Graph/completingGraph.h" 


namespace msi
{

	/**
	*	Class to generate a Small-World graph based on a mesh substrate.
	*	Supported models: beta-model, phi-model
	*
	*/
  class SW_Mesh : public Mesh {

    public:
			//!	Constructor
      SW_Mesh(const Configuration& c, unsigned int n);

			//!	Next iteration
			void nextIter();

			//! print the graph to dot file
			void print(std::ostream& o, const std::string& n) const;
      
			/**
			*		The actual Small-World generation
			*
			*	@param	p			probability
			*	@param	type	BETA|PHI
			*
			*/
			void createSmallWorld(double p, sw_type type);
			
			/**
			* True if topography is small-world graph
			* @return true if graph has small world properties, false else
			*/
			bool smallworld(){return m_isSmallWorld;}			
			
		private:

			//! Removes a randomly chosen edge
			bool removeRandomEdge();
			
			//! Rewire the edge between n1,n2 with n1 and a (random) chosen node. Returns true on success
			bool rewire(const Node *n1, const Node *n2);

			//! used by addRandomEdge()
			unsigned int insertround;
			//! used by addRandomEdge()
			unsigned int lastinsert;
			//! initial node degree for initialization
			unsigned int initialDegree;
			//!type of model to create small world graph
			std::string sw_model;
			
			//!beta parameter for small world beta-model
			double beta;
			//!phi parameter for small world beta-model
			double phi;
			
			unsigned int verbosity;
			
			//! graph has small world properties?
			bool m_isSmallWorld;		
			
			std::list<const Node*> higherDegree; // nodes with higher degree than conf.initialDegree; will loose edge
			std::list<const Node*> lowerDegree;  // nodes with lower degree than conf.initialDegree; will gain edge
  }; 
  
}  
#endif  