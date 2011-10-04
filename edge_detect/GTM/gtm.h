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


#ifndef GTM_H
#define GTM_H

#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <time.h>

#include "rbfnet.h"
#include "gmm.h"

#include "cv.h"
#include "gtm_defs.h"
#include "multi_img.h"
#include "gtm_misc.h"
#include "gtm_defs.h"
#include "neuron.h"
#include "Graph/fastDijkstra.h"
#include "Graph/completingGraph.h"
#include "edge_detection_config.h"
#include "gtm_em.h"


/** 
* Create a Generative Topographic Map.
*	Code is based on the MATLAB NETLAB implementation of Ian T Nabney
*	http://www1.aston.ac.uk/eas/research/groups/ncrg/resources/netlab/downloads/
*/
class GTM
{
  public:
		
		/**
		*	\brief Constructor
		* @param 	conf				Pointer to program config	
		* @param 	options			Struct containing parameters for function optimization, as used in NETLAB implementation
		* @param 	dim_latent	Dimensionality of the latent variables	
		* @param 	nlatent			Number of latent variables	
		* @param 	dim_data		Dimensionality of the data vectors
		* @param 	ncentres		Number of radial basis function centers
		* @param 	rbffunc			Type of radial basis function
		* @param 	prior				Gaussian zero mean prior for the parameters of the RBF model
		* @param	net					Struct containing the GTM settings
		*/
    GTM(const EdgeDetectionConfig *conf, FOptions &options, unsigned int dim_latent, unsigned int nlatent, unsigned int dim_data, unsigned int ncentres, std::string rbffunc, double prior, S_GTMNet &net);
		
		//! Destructor
    ~GTM();
		
		//! Constructor that takes a pointer to the program config and a pointer to a multispectral image
		GTM(const EdgeDetectionConfig *config, multi_img *img );
		
		/**
		*	\brief Initialise the weights and latent sample in a GTM.
		* Takes a GTM net and	generates a sample of latent data points and sets 
		* the centres (and widths if appropriate) of net.rbfnet.
		*	If the samp_type is 'REGULAR', then regular grids of latent data
		*	points and RBF centres are created. The dimension of the latent data
		*	space must be 1 or 2. 
		* The widths of the RBF basis functions are set by a call to rbfsetfwd() 
		* passing OPTIONS(7) as the scaling	parameter.
		* The RBF basis function parameters are set by a call
		*	to RBFSETBF with the data parameter as dataset and the OPTIONS
		*	vector.
		*
		*	Finally, the output layer weights of the RBF are initialised by
		*	mapping the mean of the latent variable to the mean of the target
		*	variable, and the L-dimensional latent variale variance to the
		*	variance of the targets along the first L principal components.
		*	
		* @param	net								S_GTMNET struct
		* @param	options						Struct containing parameters for function optimization, as used in NETLAB implementation
		* @param	width_factor			Width of RBF functions
		* @param	data							Data matrix, where each row contains a data vector
		* @param	samp_type					Sampling type {'REGULAR', 'UNIFORM','GAUSSIAN'}. Only REGULAR is implemented.
		* @param	latent_grid_width	Width of the regular latent grid
		* @param	rbf_hidden_units	Number of RBF centres : rbf_hidden_units x rbf_hidden_units
		*/
		void gtm_init(S_GTMNet &net, FOptions &options, double width_factor, cv::Mat1d &data, std::string samp_type, const cv::Size &latent_grid_width, const cv::Size &rbf_hidden_units );
    
		/**
		*	\brief Create the latent sample data in 2D
		*
		* @param	latent_samp_size 	Width of the regular latent grid
		* @param	latent_samp 			Matrix containing the latent points
		* @param	dim 							Dimensionality of latent variables
		*/
		void gtm_rctg(cv::Size latent_samp_size, cv::Mat1d &latent_samp, int dim);
		
		/**
		*	\brief Latent space responsibility for data in a GTM.
		*
		* @param 	net 				S_GTMNET struct
		* @param 	data 				Data matrix, where each row contains a data vector
		* @param	post 				Posterior probability P(J|X))
		* @param	activations Activations A of the GMM net.gmmnet as computed by gmmpost().
		*/		
		void gtmpost(S_GTMNet &net, cv::Mat1d &data, cv::Mat1d &post, cv::Mat1d &activations);
		
		/**
		*	\brief Probability for data under a GTM.
		*
		* @param 	net 	S_GTMNET struct
		* @param 	data 	Data matrix, where each row contains a data vector
		* @param	prob 	Probability of each point in the dataset data.
		*/
		void gtmprob(S_GTMNet &net, cv::Mat1d &data, cv::Mat1d &prob);
		
		/**
		*	\brief Mean responsibility for data in a GTM.
		*
		* @param 	net 	S_GTMNET struct
		* @param 	data 	Data matrix, where each row contains a data vector
		* @param	means The means of the responsibility distributions for each data point in	data.
		*/		
		void gtmlmean(S_GTMNet &net, cv::Mat1d &data, cv::Mat1d &means);

		//! The same as gtmlmean(), but can be used in a parallel implementation, if the loop is distributed.
		void gtmlmean_par(S_GTMNet &net, cv::Mat1d &data, cv::Mat1d &means);
		
		/**
		*	\brief Mode responsibility for data in a GTM.
		*
		* @param 	net 	S_GTMNET struct
		* @param 	data 	Data matrix, where each row contains a data vector
		* @param	modes The modes of the responsibility distributions for each data point in	data.
		*												These will always lie at one of the latent space sample points net.x
		*/			
		void gtmlmode(S_GTMNet &net, cv::Mat1d &data, cv::Mat1d &modes, cv::Mat1i &indexMaxR);
		
		//! The same as gtmlmode(), but can be used in a parallel implementation, if the loop is distributed.
		void gtmlmode_par(S_GTMNet &net, cv::Mat1d &data, cv::Mat1d &modes, cv::Mat1i &indexMaxR);
		
		/**
		*	\brief Magnification factors for a GTM.
		*
		*	Computes the magnification factors for each point the latent space
		* contained in net.x.
		* @param 	net 						S_GTMNET struct
		* @param	magnifications 	Matrix, where the calculated magnification factors are stored.
		*/		
		void gtmmag(S_GTMNet &net, cv::Mat1d &magnifications);

		/**
		*	\brief Computes positions in the responsibility matrix with maximum responsibility for data in a GTM.
		*
		* @param	net 			S_GTMNET struct
		* @param	R 				Responsibility matrix, where each row contains the posterior probability for the latetn variables for 
		*														a specific data point
		* @param	maxIndex 	The index inside each row of R that contains the largest responsibility value.
		*/					
		void gtmmaxresp(S_GTMNet &net, cv::Mat1d &R, cv::Mat1i &maxIndex);

		/**
		*	\brief Computes the latent variables with maximum responsibility for data in a GTM.
		*
		* @param	net 			S_GTMNET struct
		* @param	ndata 		S_GTMNET struct
		* @param	R 				Responsibility matrix, where each row contains the posterior probability for the latetn variables for 
		*										a specific data point
		* @param	maxIndex 	The index inside each row of R that contains the largest responsibility value.
		*/		
		void gtmmaxmode(S_GTMNet &net,cv::Mat1i &maxIndex, cv::Mat1d &modes);
// 		void gtmmaxmode(S_GTMNet &net, unsigned int ndata,cv::Mat1i &maxIndex, cv::Mat1d &modes);

		/**
		*	\brief Computes distance maps by evaluating the distances of the means
		*
		* @param	means 		Vector containing the means for each data point
		* @param	distanceX Horizontal distance map
		* @param	distanceY Vertical distance map
		* @param	mode 			0: Sobel, 1: Scharr filtering
		*/
		void getEdgeByMeans(cv::Mat1d &means, cv::Mat1d &distanceX, cv::Mat1d &distanceY , int mode=0);
		
		/**
		*	\brief Computes distance maps by evaluating the distances of the means
		*
		* @param	modes 		Vector containing the means for each data point
		* @param	distanceX Horizontal distance map
		* @param	distanceY Vertical distance map
		* @param	mode 			0: Sobel, 1: Scharr filtering
		*/		
		void getEdgeByModes(cv::Mat1d &modes, cv::Mat1d &distanceX, cv::Mat1d &distanceY , int mode=0);
		
		/**
		*	\brief Adds the magnification factors for each latent (x,y) pair as a third dimension
		*
		* @param	latentMap 			2-dimensional vector containing x-coordinates of the latent variable in latentMap[0] and 
		*																	y-coordinates of the latent variable in latentMap[1]. The vector is extended by a third plane, containing the
																			magnification factors
		* @param	magnifications  matrix containin the magnicifation factors.
		*/				
		void mapMagnifications(std::vector<cv::Mat> &latentMap, cv::Mat1d &magnifications );
		
		//! Returns a string containing the concatenated used parameter settings.
		std::string getFileExtension();
		
		//! Initialises a weighted graph, where the nodes are latetn grid points. The edge weights are calculated using magnifications
		void initDijkstra(cv::Mat1d &magnifications);

		/**
		*	\brief Computes the shortest path between node (index1Y,index1X) and (index2Y,index2X)
		*
		* @param	index1Y 				Y- coordinate of first node
		* @param 	index1X 				X- coordinate of first node
		* @param	index2X 				Y- coordinate of second node
		* @param	index2Y 				X- coordinate of second node
		* @param	useEdgeWeigths 	True, if the magnification factors are used for distance calculation.
		*/
		double graphDistance(unsigned int index1Y,unsigned int index1X, unsigned int index2Y,unsigned int index2X, bool useEdgeWeigths=true);

		/**
		*	\brief Computes the shortest path between node index1 and index2
		*
		* @param	index1 					Index of first node
		* @param	index2 					Index of second node
		* @param	useEdgeWeigths 	True, if the magnification factors are used for distance calculation.
		*/
		double graphDistance(unsigned int index1, unsigned int index2, bool useEdgeWeigths=true);
		
		/**
		*	\brief Computes the shortest path between two (non-integer) points in the latent space
		*
		* @param	p1 	Coordinates of p1
		* @param	p2	Coordinates of p2
		*/		
		double graphDistance(cv::Point2d p1, cv::Point2d p2);
		
		//! Starts GTM algorithm
		void execute();

		//! Returns width of latent grid
		unsigned int latentWidth(){return m_latentWidth;}
		
		//! Returns height of latent grid
		unsigned int latentHeight(){return m_latentHeight;}
		
		//! Computes the edge weights according dissimilarities of the magnification factors
		void umatrix(cv::Mat1d &magnifications);
	    
	private:
		unsigned int m_MSIwidth;
		unsigned int m_MSIheight;
		
		unsigned int m_latentWidth;
		unsigned int m_latentHeight;
		
		cv::Mat1d latentXCoords;
		cv::Mat1d latentYCoords;
		
		const EdgeDetectionConfig *m_conf;
		multi_img *m_msi;
		
		S_GTMNet m_gtmnet;

		msi::Mesh *m_mesh;
		cv::Mat1d m_distanceMap;
		cv::Mat1i m_maxModes;
		FOptions m_options;
		
		bool withGraph;
		bool withUMap;
		
		std::stringstream infoStream;
};

#endif 
