#ifndef NEURON_H
#define NEURON_H

#include <multi_img.h>
#include <cv.h> // opencv
#include <vector> // std library
#include <cmath>

class Neuron : public multi_img::Pixel {

public:
	Neuron() : multi_img::Pixel() {}

	Neuron(int dimension) : multi_img::Pixel(dimension) {}

	Neuron(const multi_img::Pixel &n) : multi_img::Pixel(n) {}

	/**
	* Uniformly randomizes the neurons multi_img::Value values
	* within the given range.
	*
	* @param	lower Lower bound of the uniform range
	* @param	upper Upper bound of the uniform range
	*/
	void randomize(cv::RNG &rng, multi_img::Value lower, multi_img::Value upper)
	{
		// 1D temporary array with the size of this neuron to store randomized values
		cv::Mat_<multi_img::Value> target(*this);
		rng.next();
		rng.fill(target, cv::RNG::UNIFORM, cv::Scalar(lower), cv::Scalar(upper));
	}

	/** TODO: replace with similarity measure!
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
};


#endif // NEURON_H
