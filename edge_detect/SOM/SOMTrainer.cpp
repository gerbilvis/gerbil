#include "SOMTrainer.h"

SOMTrainer::SOMTrainer(void)
{
}

SOMTrainer::~SOMTrainer(void)
{
}

void SOMTrainer::setParameters(double dLearnRateStart, double dLearnRateEnd, double dAdaptionStart, double dAdaptionEnd, unsigned int uiMaxIterations)
{
	//set parameters
	m_dLearnRateStart = dLearnRateStart;
	m_dLearnRateEnd = dLearnRateEnd;
	m_dAdaptionRadiusStart = dAdaptionStart;
	m_dAdaptionRadiusEnd = dAdaptionEnd;
	m_uiMaxIteration = uiMaxIterations;

	//reset iteration counter
	m_uiCurrentIteration = 0;
}

void SOMTrainer::iterate(SelfOrganisingMap &map, double c1, double c2)
{
	//adjust iteration parameters
	double learnRate = m_dLearnRateStart*powl(m_dLearnRateEnd/m_dLearnRateStart, (double)m_uiCurrentIteration/(double)m_uiMaxIteration);
	double adaptionRadius = m_dAdaptionRadiusStart*powl(m_dAdaptionRadiusEnd/m_dAdaptionRadiusStart, (double)m_uiCurrentIteration/(double)m_uiMaxIteration);

	//get winning neuron
	Position winner;
	map.getWinner(c1, c2, &winner);

	//calculate affected neuron positions
	int minX = 0;
	int minY = 0;
	int maxX = map.getWidth()-1;
	int maxY = map.getHeight()-1;
	if (winner.x - (int)adaptionRadius > minX)
		minX = winner.x - (int)adaptionRadius;
	if (winner.y - (int)adaptionRadius > minY)
		minY = winner.y - (int)adaptionRadius;
	if (winner.x + (int)adaptionRadius < maxX)
		maxX = winner.x + (int)adaptionRadius;
	if (winner.y + (int)adaptionRadius < maxY)
		maxY = winner.y + (int)adaptionRadius;

	//adjust neuron codebook vectors
	Neuron* cNeuron;
	for (int i = minX; i <= maxX; ++i) {
		for (int j = minY; j <= maxY; ++j) {
			cNeuron = map.get(i, j);

			//calculate distance (squared) of current neuron from winning neuron
			double dx = (double)winner.x-(double)i;
			double dy = (double)winner.y-(double)j;
			double sq = dx*dx + dy*dy;

			//adjust only if current Neuron is within adaptionradius
			if (sq <= adaptionRadius*adaptionRadius) {
				//calculate the factor to push current neuron against current training sample
				double learnFactor = learnRate*exp(-sq/(2.0*adaptionRadius*adaptionRadius));

				//push current neuron against current training sample
				cNeuron->c1 = cNeuron->c1 + learnFactor*(c1-cNeuron->c1);
				cNeuron->c2 = cNeuron->c2 + learnFactor*(c2-cNeuron->c2);
			}
		}
	}

	//increment iteration counter
	++m_uiCurrentIteration;
}

void SOMTrainer::calibrate(SelfOrganisingMap &map, double c1, double c2, bool positiveLabel)
{
	//get winning neuron
	Neuron* neuron = map.getWinner(c1, c2, NULL);

	//if this training sample has "positive" label increment label counter
	if (positiveLabel) neuron->l += 1;
	//else decrement label counter
	else neuron->l -= 1;
}

void SOMTrainer::reset()
{
	m_uiCurrentIteration = 0;
}
