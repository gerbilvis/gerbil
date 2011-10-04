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


#ifndef GTM_DEFS_H
#define GTM_DEFS_H

#include "gtm.h"
#include "rbfnet.h"
#include "gmm.h"

/** 
* Struct for constraint optimization on functions as used in 
* MATLAB NETLAB implementation
*/
class GTM;

/**	
*	\struct FOptions
*	Struct for function optimization as used in MATLAB NETLAB implementation 
*	as used in NETLAB implementation
*
*/
struct FOptions
{
	int display_parameter;												//!\brief	Display parameter (Default:0). 1 displays some results.
	double termination_tolerance_x;								//!\brief	Termination tolerance for X.(Default: 1e-4).
	double termination_tolerance_F;								//!\brief	Termination tolerance on F.(Default: 1e-4)
	double termination_constraint_criterion;			//!\brief	Termination criterion on constraint violation.(Default: 1e-6)
	std::string algorithm_strategy;								//!\brief	Algorithm: Strategy:  Not always used.
	std::string algorithm_optimizer;							//!\brief	Optimizer: Not always used. 
	bool algorithm_line_search;										//!\brief	Line Search Algorithm. (Default 0)
	double function_value_lambda;									//!\brief	Function value. (Lambda in goal attainment. )
	bool user_supplied_gradients;									//!\brief	Set to 1 if you want to check user-supplied gradients.
	unsigned int number_function_constraint_eval; //!\brief	Number of Function and Constraint Evaluations.
	unsigned int number_function_gradient_eval;		//!\brief	Number of Function Gradient Evaluations.
	unsigned int number_constraint_eval;					//!\brief	Number of Constraint Evaluations.
	unsigned int number_equality_constraints;			//!\brief	Number of equality constraints. 
	unsigned int max_number_function_eval;				//!\brief	Maximum number of function evaluations. (Default is 100*number of variables)

	unsigned int unknown_variable;								//!\brief	Used in goal attainment for special objectives. 
	double delta_min;															//!\brief	Minimum change in variables for finite difference gradients.
	double delta_max;															//!\brief	Maximum change in variables for finite difference gradients.
	double step_length;														//!\brief	Step length. (Default 1 or less).
	
};

typedef enum eemverbosity
{
	HIGH 	= 0,
	ON 		=	1,	
	OFF 	=	2
}EEmVerbosity;


/*!
*	\struct S_GTMNet
*	Struct that handles the Generative Topographic Mapping model
*/ 
struct S_GTMNet
{
	std::string type;					//!\brief	Type of struct ('gtm')
	unsigned int nin;					//!\brief	Number of input units for RBF net
	unsigned int dim_latent;	//!\brief	Dimensionality of latent variables
	S_RBFNet rbfnet;					//!\brief	Struct for RBFNET
	RBFNet *rbf;							//!\brief	Pointer to RBFNET instance
	S_GMMNet gmmnet;					//!\brief	Struct for GMMNET
	GMM *gmm;									//!\brief	Pointer to GMMNET instance
	GTM *gtm;									//!\brief Pointer to GTMNET instance
	cv::Mat1d X;							//!\brief Latent grid
};

#endif
