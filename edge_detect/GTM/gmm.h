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

#ifndef GMM_H
#define GMM_H 
 
#include <string>
#include "cv.h"
#include "gtm_misc.h"
 


 
/*!\struct S_GMMNet
*		struct that handles the Gaussian Mixture Model
*		as used in NETLAB implementation
*/
struct S_GMMNet
{
	std::string type;					//!\brief Type of struct ('gmm')
	unsigned int nin;					//!\brief The dimension of the space
	unsigned int ncentres;		//!\brief Number of mixture components
	std::string covar_type;		//!\brief Type of covariance model
	unsigned int nwts;				//!\brief Total number of weights and biases of the RBF net
	cv::Mat1d priors;					//!\brief Mixing coefficients
	cv::Mat1d centres;				//!\brief Means of Gaussians: stored as rows of a matrix
	cv::Mat1d covars;					//!\brief Covariances of Gaussians
};

/** 
* Creates a Gaussian mixture model with specified architecture.
*	Code is based on the MATLAB NETLAB implementation of Ian T Nabney
*	http://www1.aston.ac.uk/eas/research/groups/ncrg/resources/netlab/downloads/
*/
class GMM
{
	public:
		/**
		* \brief Constructor
		* @param	dim					Data dimensionality
		* @param	ncentres		Number of centres in the mixture model
		* @param	covar_type	Type of mixture model : {'SPHERICAL'|'DIAG'(not implemented)|'FULL'(not implemented)|'PPCA'(not implemented)}
		* @param	net					S_GMMNET struct
		* @param	fixedSeed		If true, random generators are initialized with a fixed seed
		*/
		GMM(unsigned int dim, unsigned int ncentres, std::string covar_type, S_GMMNet &net, bool fixedSeed=false);
		//! Destructor
		~GMM();

		//! Prints the 	contents of the S_GMMNet
		void dumpNet(S_GMMNet &net);
		
		/*!\brief Computes the class posterior probabilities of a Gaussian mixture model.
		*		
		*	Description
		*	This function computes the posteriors POST (i.e. the probability of
		*	each component conditioned on the data P(J|X)) for a Gaussian mixture
		*	model.
		*
		* @param[in] 			mix 					S_GMMNET struct
		* @param[in] 			x 						Each row of x contains a data vector
		* @param[in,out] 	post 					Posterior probability P(J|X))
		* @param[in,out] 	activations 	Activations that are returned by gmmactiv()
		*/	
		void gmmpost(S_GMMNet &mix,cv::Mat1d &x,cv::Mat1d &post,cv::Mat1d &activations);

		
		/*!\brief Computes the activations of a Gaussian mixture model.
		*		
		*	Description
		*	This function computes the activations A (i.e. the  probability
		*	P(X|J) of the data conditioned on each component density)  for a
		*	Gaussian mixture model.
		*
		* @param[in]			mix 	S_GMMNET struct
		* @param[in]			x 		Data vectors
		* @param[in,out]	a 		Activations 
		*/	
		void gmmactiv(S_GMMNet &mix,cv::Mat1d &x, cv::Mat1d &a);
		
		/*!\brief Computes the data probability for a Gaussian mixture model.
		*		
		*	Description
		*	This function computes the unconditional data density P(X) for a
		*	Gaussian mixture model.
		*
		* @param[in]			mix 	S_GMMNET struct
		* @param[in]			x 		Data vectors
		* @param[in,out]	prob 	Unconditional density P(X)
		*/		
		void gmmprob(S_GMMNet &mix,cv::Mat1d &data, cv::Mat1d &prob);
};

#endif
 
