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


#include <cassert>

#include "neuron.h"
#include "cv.h"

void Neuron::randomize(cv::RNG &rng ,multi_img::Value lower, multi_img::Value upper) 
{
	// 1D temporary array with the size of this neuron to store randomized values
	cv::Mat_<multi_img::Value> randValues( 1, (*this).size());
	rng.next();
	rng.fill(randValues, cv::RNG::UNIFORM, cv::Scalar(lower), cv::Scalar(upper) );
	
	// assigning generated values
	for(unsigned int i = 0; i < (*this).size(); i++) {
		(*this)[i] = randValues[0][i];
	}
}
