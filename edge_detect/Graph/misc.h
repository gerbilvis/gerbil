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


#ifndef MISC_H
#define MISC_H

#include <cstdlib>
#include <vector>
#include <time.h>
#include <string>
#include <sstream>
#include <limits.h>


namespace msi
{
	/**
	*	\enum InsertionMode
	*
	*/
  enum InsertionMode
  {
    UNDEFINED =0,
    OTHER
  };

	/**
	*	\struct Configuration
	*	Contains all properties needed for the graph setup
	*
	*/
  struct Configuration
  {
    bool directed;							//!	\brief True, if graph is supposed to be directed
    bool completing;						//!	\brief True, if graph is supposed to be a completing graph
    bool periodic;							//!	\brief True, if graph is supposed to be periodic
    unsigned int nodes;					//!	\brief Number of nodes
    InsertionMode insertion;		//!	\brief Insertion mode
    unsigned int startcomp;			//!	\brief Iteration where completing starts
    unsigned int finishcomp;		//!	\brief Iteration where completing ends
    unsigned int initialDegree;	//!	\brief Degree of each node ( may differ at border nodes, if graph is non-periodic)
    std::string graph_type;			//!	\brief 'MESH' for basic 2D mesh, 'MESH_P' for periodic 2D mesh
    std::string sw_model;				//!	\brief 'BETA'|'PHI'
    double beta;								//!	\brief Value for beta
    double phi;									//!	\brief Value for phi
		unsigned int width;					//!	\brief Width of graph
		unsigned int height;				//!	\brief Height of graph
    unsigned int maxIter;				//!	\brief Number of training iterations
		std::string output;					//!	\brief Filename for .dot file
  };

	/**
	*	\enum sw_type
	*
	*/	
  enum sw_type
  {
    BETA=0,		
    PHI
  };
  
	//! Return true if x is a power of 2
  inline bool checkPowerOfTwo(unsigned int x)
  {
    return ( ((x & (x-1)) == 0) && (x > 0) );
  }
  
  //! Return log2(x)
  inline unsigned int log2(unsigned int x)
  {
    unsigned int res = 0;
    if(x == 0)return UINT_MAX;
    while(x != 1)
    {
      x>>=1;
      res++;
    }  
    return res;
  }

}

/**
*	Class for an RGB triplet
*	Used for the visualization of the graph structure
*/
struct Color 
{
    Color(unsigned int r, unsigned int g, unsigned int b);
    Color(unsigned int c);
    Color();
    virtual ~Color();
    
    std::string rgb2hex();
    std::string rgb2string();

    unsigned int red,green,blue;
};

#endif
