#include "SelfOrganisingMap.h"

SelfOrganisingMap::SelfOrganisingMap(unsigned int uiWidth, unsigned int uiHeight)
{
	m_uiWidth = uiWidth;
	m_uiHeight = uiHeight;

	//allocate neuron array
	m_ppNeurons = new Neuron*[m_uiWidth];
	for (unsigned int i = 0; i < m_uiWidth; ++i) {
		m_ppNeurons[i] = new Neuron[m_uiHeight];
		for (unsigned int j = 0; j < m_uiHeight; ++j) {
			m_ppNeurons[i][j].c1 = 0.0;
			m_ppNeurons[i][j].c2 = 0.0;
			m_ppNeurons[i][j].l = 0;
		}
	}
	
}

SelfOrganisingMap::~SelfOrganisingMap(void)
{
	//delete neuron array
	for (unsigned int i = 0; i < m_uiWidth; ++i) {
		delete[] m_ppNeurons[i];
	}
	delete[] m_ppNeurons;
}

unsigned int SelfOrganisingMap::getWidth()
{
	return m_uiWidth;
}

unsigned int SelfOrganisingMap::getHeight()
{
	return m_uiHeight;
}

void SelfOrganisingMap::randomize()
{
	//randomize codebook vectors
	srand((unsigned int)time(0));
	for (unsigned int i = 0; i < m_uiWidth; ++i) {
		for (unsigned int j = 0; j < m_uiHeight; ++j) {
			//values between 0.0 and 1.0
			m_ppNeurons[i][j].c1 = fabs((double)rand()/(double)RAND_MAX);
			m_ppNeurons[i][j].c2 = fabs((double)rand()/(double)RAND_MAX);
		}
	}
}

Neuron* SelfOrganisingMap::get(unsigned int x, unsigned int y)
{
	//ensure x and y is in valid range
	assert(x < m_uiWidth && y < m_uiHeight);
	if (x >= m_uiWidth || y >= m_uiHeight)
		return NULL;

	return &m_ppNeurons[x][y];
}

Neuron* SelfOrganisingMap::getWinner(double c1, double c2, Position* pos)
{
	unsigned int x = 0;				//x position of winning neuron
	unsigned int y = 0;				//y position of winning neuron
	double sqDistance = DBL_MAX;	//minimum distance, init with DOUBLE MAX_VALUE

	//iterate over neurons
	for (unsigned int i = 0; i < m_uiWidth; ++i) {
		for (unsigned int j = 0; j < m_uiHeight; ++j) {
			double d1 = (double)m_ppNeurons[i][j].c1-(double)c1;	
			double d2 = (double)m_ppNeurons[i][j].c2-(double)c2;
			double sd = d1*d1 + d2*d2;
			//if current distance (squared) is smaller
			//set this as new minimum
			if (sd < sqDistance) {
				sqDistance = sd;
				x = i;
				y = j;
				if (sd == 0.0) {
					if (pos != NULL) {
						pos->x = x;
						pos->y = y;
					}
					return &m_ppNeurons[x][y];
				}
			}
		}
	}

	//if position struct is given, assign values
	if (pos != NULL) {
		pos->x = x;
		pos->y = y;
	}

	return &m_ppNeurons[x][y];
}

bool SelfOrganisingMap::loadFromFile(SelfOrganisingMap **map, const char* cpFile)
{
	//open file stream
	std::fstream stream;
	stream.open(cpFile, std::ios_base::in | std::ios_base::binary);

	if (!stream.is_open())
		return false;

	unsigned int height;
	unsigned int width;

	//read width and height from file
	stream.read((char*)&width, sizeof(unsigned int));
	stream.read((char*)&height, sizeof(unsigned int));

	//create SOM
	*map = new SelfOrganisingMap(width, height);

	//read neuron values from file
	for (unsigned int i = 0; i < (*map)->m_uiWidth; ++i) {
		for (unsigned int j = 0; j < (*map)->m_uiHeight; ++j) {
			stream.read((char*)&(*map)->m_ppNeurons[i][j].c1, sizeof(double));
			stream.read((char*)&(*map)->m_ppNeurons[i][j].c2, sizeof(double));
			stream.read((char*)&(*map)->m_ppNeurons[i][j].l, sizeof(int));
		}
	}

	return true;
}

bool SelfOrganisingMap::saveToFile(SelfOrganisingMap *map, const char *cpFile)
{
	//open file stream
	std::fstream stream;
	stream.open(cpFile, std::ios_base::out | std::ios_base::binary);

	if (!stream.is_open())
		return false;

	//write width and height to file
	stream.write((char*)&map->m_uiWidth, sizeof(unsigned int));
	stream.write((char*)&map->m_uiHeight, sizeof(unsigned int));

	//write neuron values to file
	for (unsigned int i = 0; i < map->m_uiWidth; ++i) {
		for (unsigned int j = 0; j < map->m_uiHeight; ++j) {
			stream.write((char*)&map->m_ppNeurons[i][j].c1, sizeof(double));
			stream.write((char*)&map->m_ppNeurons[i][j].c2, sizeof(double));
			stream.write((char*)&map->m_ppNeurons[i][j].l, sizeof(int));
		}
	}

	stream.close();
	return true;
}

