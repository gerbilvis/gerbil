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


#ifndef GTM_EM_H
#define GTM_EM_H

#include <sstream>

#include "gtm_misc.h"
#include "gtm.h"
#include "gtm_defs.h"
#include "edge_detection_config.h"

/**
*
*	Uses the Expectation Maximization algorithm to estimate the parameters of a GTM defined by
*	a data structure net. The matrix T represents the data whose expectation is maximized, with each row corresponding to a vector.
* Code is based on the MATLAB NETLAB implementation of Ian T Nabney
*	http://www1.aston.ac.uk/eas/research/groups/ncrg/resources/netlab/downloads/
*
*/
class EM
{
	public:
		
		/**
		*	\brief Constructor
		* @param 	config	Program config	
		* @param 	net			Struct containing the GTM settings
		* @param 	t				Data matrix, where each row contains a data vector
		* @param 	options	Struct containing parameters for function optimization, as used in NETLAB implementation
		* @param 	store		If true, the values of the error function is stored
		* @param	net			Struct containing the GTM settings
		*/		
		EM(const EdgeDetectionConfig *config ,S_GTMNet &net, cv::Mat1d &t, FOptions &options , bool store=true);
		//! Destructor
		~EM();
		//! Return string containing the used parameter
		std::string getErrlog(){return infoStream.str();}
	
	private:
		
		std::stringstream infoStream;
		cv::Mat1d errlog;
		
		unsigned int m_niters;
		
		bool m_store;
		bool m_test;
		
		const EdgeDetectionConfig *m_config;
		EEmVerbosity m_verbosity;
};
	

#endif
