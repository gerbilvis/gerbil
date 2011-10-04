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


#ifndef RBFNET_H
#define RBFNET_H 
 
#include <string>
#include <cmath>
#include <vector>
#include "cv.h"
#include "gtm_misc.h"
 


/*!
*	\struct S_RBFNet
*	Struct that handles the Radial Basis Function network model 
*	as used in NETLAB implementation
*/ 
struct S_RBFNet
{
	std::string type;			//!\brief Type of struct ('rbf')
	unsigned int nin;			//!\brief Number of input units
	unsigned int nhidden;	//!\brief Number of hidden units
	unsigned int nout;		//!\brief Number of output units
	unsigned int nwts;		//!\brief Total number of weights and biases
	double alpha;					//!\brief Corresponding to a zero-mean isotropic Gaussian prior	with inverse variance
	std::string actfn;		//!\brief String defining hidden unit activation function: {'GAUSSIAN'|'TPS'|'R4LOGR'}
	std::string outfn;		//!\brief String defining hidden unit activation function: {'LINEAR'|'Neuroscale' (not implemented)}
	cv::Mat1d c;					//!\brief Centres
	cv::Mat1d wi;					//!\brief Squared widths (null, for TPS and R4LOGR)
	cv::Mat1d w2;					//!\brief Second layer weight matrix
	cv::Mat1d b2;					//!\brief Second Layer bias vector
	cv::Mat1d mask;				//!\brief Mask selects only output layer weights
	};

/** 
* Creates an RBF network with specified architecture
* Code is based on the MATLAB NETLAB implementation of Ian T Nabney.
*	http://www1.aston.ac.uk/eas/research/groups/ncrg/resources/netlab/downloads/
*/	
class RBFNet
{
	public:
	
	/**
	*	\brief Constructor
	*
	* @param 			nin			Number of input units
	* @param 			nhidden	Number of hidden units
	* @param 			nout		Number of output units
	* @param 			func		String defining hidden unit activation function {'GAUSSIAN', 'TPS', 'R4LOGR'}
	* @param 			prior		Corresponding to a zero-mean isotropic Gaussian prior	with inverse variance with value prior
	* @param 			net			Struct containing the RBF settings
	*/	
	RBFNet(unsigned int nin, unsigned int nhidden, unsigned int nout, std::string func, double prior, S_RBFNet &net);
	//! Destructor
	~RBFNet();
    
	/**
	* Set basis function widths of RBF.
	*
	* @param	net 		Struct containing the RBF settings
	* @param	scale 	If Gaussian basis functions are used, then the variances are set to the largest squared distance between 
	* 								centres if scale is non-positive and scale times the mean distance of each centre to its nearest neighbour 
	*									if scale is positive. Non-Gaussian basis functions do not have a width.
	*/
	void rbfsetfw(S_RBFNet &net, double scale );
    
	/**
	* Forward propagation
	*
	* @param	net				Struct containing the RBF settings
	* @param	x 				Matrix of input vectors
	* @param	outputs		Matrix of output vectors
	* @param	phi				matrix of hidden unit activations
	* @param	distances	The squared distances between each basis function centre and each pattern in which each row corresponds to a data point.
	*
	*/
	void rbffwd(S_RBFNet &net, cv::Mat1d &x, cv::Mat1d &outputs, cv::Mat1d &phi, cv::Mat1d &distances );
    
	/**
	* Evaluate derivatives of RBF network outputs with respect to inputs.
	*
	* @param 	net 				Struct containing the RBF settings
	* @param	latent_data Matrix containing the latent variables
	* @param 	jac 				The evaluated Jacobian for each latent point
	*/
	void rbfjacob(S_RBFNet &net, cv::Mat1d &latent_data, std::vector<cv::Mat1d> &jac);
    

	//! Packs all parameters in the network net into one weight parameter w
	void rbfpak(S_RBFNet &net, cv::Mat1d &w);
    
	
	//! Inversion of rbfpak
	void rbfunpak(S_RBFNet &net, cv::Mat1d w);
    

	//! Display the content of the net structure
	void dumpNet(S_RBFNet &net);
  
	/**
	* Create Gaussian prior and output layer mask for RBF.
	*
	*	@param	rbfunc	String defining hidden unit activation function {'GAUSSIAN', 'TPS', 'R4LOGR'}
	*	@param	nin			Number of input units
	*	@param	nhidden	Number of hidden units
	*	@param	nout		Number of output units
	*	@param	net			Struct containing the RBF settings
	*/
	void rbfprior(std::string rbfunc, unsigned int nin, unsigned int nhidden, unsigned int nout, S_RBFNet &net);
    
	private:

	cv::Mat1d c;

	cv::Mat1d wi;

	cv::Mat1d w2;

	cv::Mat1d b2;
    
	double m_prior;
	double alpha;

	std::string outfunc; 
  
 };
 #endif
 