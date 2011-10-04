/**
 * @file SOMTrainer.h
 * @brief Training of the Self-Organizing Maps,
 * @author Christoph Malskies
 * @date January 2010 
 */

#ifndef SOMTRAINER_H
#define SOMTRAINER_H

#include "SelfOrganisingMap.h"

class SOMTrainer
{
public:
	SOMTrainer(void);
	~SOMTrainer(void);

	/**
		Sets the training parameters.
		The learn rate and adaptionradius is decremented monotonically during training process.
		@param dLearnRateStart Learning rate at the beginning.
		@param dLearnRateEnd Learning rate at the end.
		@param dAdaptionStart Adaption radius at the beginning.
		@param dAdaptionEnd Adaption radius at the end.
		@param uiMaxIterations Number of iterations to be performed.
	*/
	void setParameters(double dLearnRateStart, double dLearnRateEnd, double dAdaptionStart, double dAdaptionEnd, unsigned int uiMaxIterations);

	/**
		Performs one iteration step.
		The internal iteration counter is incremented and the learning parameters are adjusted.
		@param map The SOM on which the training takes place.
		@param c1 First value of training sample.
		@param c2 Second value of training sample.
	*/
	void iterate(SelfOrganisingMap &map, double c1, double c2);

	/**
		Performs one calibration step.
		The winning neuron will be detected and its label counter will be incremented or decremented according to the given label.
		@param map The SOM in which the calibration takes place.
		@param c1 First value of calibration sample.
		@param c2 Second value of calibration sample.
		@param positiveLabel True if label counter should be incremented, fals otherwise.
	*/
	void calibrate(SelfOrganisingMap &map, double c1, double c2, bool positiveLabel);

	/**
		Resets the internal iteration counter.
	*/
	void reset();

private:
	unsigned int m_uiMaxIteration;
	unsigned int m_uiCurrentIteration;
	double m_dLearnRateStart;
	double m_dLearnRateEnd;
	double m_dAdaptionRadiusStart;
	double m_dAdaptionRadiusEnd;
};

#endif //SOMTRAINER_H
