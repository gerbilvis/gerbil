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


#ifndef NEURON_H
#define NEURON_H

#include <multi_img.h>
#include "cv.h" // opencv
#include <vector> // std library
#include <cmath>

class Neuron : public multi_img::Pixel {

public:

	/**
	* Simple Neuron constructor initializes multi_img::Value vector
	* with a size of zero.
	*/
	Neuron() : multi_img::Pixel(), marked(false) {}

	/**
	* Neuron constructor initializes multi_img::Value vector
	* with the specified size.
	*
	* @param	dimension Size of vector
	*/
	Neuron(int dimension) : multi_img::Pixel(dimension), marked(false) {}

	/**
	* Neuron constructor initializes multi_img::Value vector
	* with the given vector and copies its values.
	*
	* @param	n std::vector
	*/
	Neuron(const multi_img::Pixel &n) : multi_img::Pixel(n), marked(false) {}

	/**
	* Uniformly randomizes the neurons multi_img::Value values
	* within the given range.
	*
	* @param	lowerBound Lower bound of the uniform range
	* @param	upperBound Upper bound of the uniform range
	*/
// 	void randomize(multi_img::Value lowerBound, multi_img::Value upperBound);
	
	/**
	* Uniformly randomizes the neurons multi_img::Value values
	* within the given range.
	*
	* @param	lowerBound Lower bound of the uniform range
	* @param	upperBound Upper bound of the uniform range
	*/
	void randomize(cv::RNG &rng, multi_img::Value lowerBound, multi_img::Value upperBound);	

	/**
	* Calculates the euclidean norm of this vector
	*
	* @return	Euclidean norm
	*/
	inline double euclideanNorm() {
		double ret = 0.;

		for (unsigned int i = 0; i < (*this).size(); i++) {
			ret += (*this)[i] * (*this)[i];
		}
		return std::sqrt(ret);
	}

	/**
	* Calculates the euclidean norm of the difference
	* between this Neuron and the given one.
	*
	* @param	n Other neuron to calculate distance
	* @return	Euclidean distance
	*/
	inline double euclideanDistance(const multi_img::Pixel &n) {
		assert(n.size() == size());
		double ret = 0.;

		for (unsigned int i = 0; i < size(); i++) {
			ret += ((*this)[i] - n[i]) * ((*this)[i] - n[i]);
		}
		// take the squared distance or root squared distance
		return std::sqrt(ret);
	}

	/**
	  * Update vector by shifting it to new vector with a weight,
	  * this = this + (input - this)*weight;
	  */
	void update(const multi_img::Pixel input, double weight) {
		// let OpenCV do the work for us
		cv::Mat_<multi_img::Value> in(input, false);
		cv::Mat_<multi_img::Value> out(*this, false);
		out += (in - out)*weight;
	}

	//! Return RGB triplet (used at graph visualization)
	cv::Vec3f getRGB(){return m_rgb;}

	//! Return BGR triplet (used for graph visualization)
	inline cv::Vec3f getBGR()
	{
		cv::Vec3f inv;
		inv[0] = m_rgb[2];
		inv[1] = m_rgb[1];
		inv[2] = m_rgb[0];
		return inv;
  }
    
  //! Set RGB triplet (used for graph visualization)
  inline void setSRGB(cv::Vec3f v)
  {
    m_rgb =v;
  }
    
	bool marked;

private:
    cv::Vec3f m_rgb;
		                                  
};


#endif // NEURON_H
