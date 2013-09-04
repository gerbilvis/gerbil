/*	
	Copyright(c) 2012 Ralph Muessig	and Johannes Jordan
	<johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef NEURON_H
#define NEURON_H

#include <multi_img.h>
#include <vector>
#include <cmath>

class Neuron : public multi_img::Pixel {

public:
	Neuron() : multi_img::Pixel() {}

	Neuron(size_t dimension) : multi_img::Pixel(dimension) {}

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
		cv::Mat_<multi_img::Value> target(*this);
		rng.next();
		rng.fill(target, cv::RNG::UNIFORM, cv::Scalar(lower), cv::Scalar(upper));
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
