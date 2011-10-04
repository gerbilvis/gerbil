/**
 * @file SelfOrganisingMap.h
 * @brief Main part of the Self-Organizing Maps,
 * @author Christoph Malskies
 * @date January 2010 
 */


#ifndef SELFORGANISINGMAP_H
#define SELFORGANISINGMAP_H

#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "assert.h"
#include "float.h"

#include <iostream>

#include "defines.h"

class SelfOrganisingMap
{
public:
	SelfOrganisingMap(unsigned int uiWidth, unsigned int uiHeight);
	~SelfOrganisingMap(void);

	/**
		Returns the width of the SOM.
		@return Number of Neurons (Width)
	*/
	unsigned int getWidth();
	/**
		Returns the height of the SOM.
		@return Number of Neurons (Height)
	*/
	unsigned int getHeight();

	/**
		Randomizes the codebook vectors of the Neurons.
		The random values lie between 0.0 and 1.0.
	*/
	void randomize();

	/**
		Retrieves a pointer to the Neuron at position (x,y).
		If x or y is outside the valid range (width,height), NULL is returned.
		@param x X Position of requested Neuron.
		@param y Y Position of requested Neuron.
		@return Neuron at position (x,y).
	*/
	Neuron* get(unsigned int x, unsigned int y);
	/**
		Returns the winning Neuron according to the given codebook values.
		The position of the winning Neuron is stored in "pos". "pos" can be set to NULL.
		@param c1 First value of sample.
		@param c2 Second value of sample.
		@param pos [optional] The position of the winning Neuron is stored here.
		@return Winning Neuron.
	*/
	Neuron* getWinner(double c1, double c2, Position* pos);

	/**
		Loads a SOM from the given file in binary format.
		@param map The loaded SOM is stored here.
		@param cpFile Name of SOM file.
		@return True on success, false on error.
	*/
	static bool loadFromFile(SelfOrganisingMap **map, const char* cpFile);
	/**
		Stores a given SOM to a given file in binary format.
		@param map The map which should be stored.
		@param cpFile Name of SOM file.
		@return True on success, false on error.
	*/
	static bool saveToFile(SelfOrganisingMap *map, const char* cpFile);

private:
	unsigned int m_uiWidth;
	unsigned int m_uiHeight;

	Neuron** m_ppNeurons;
};

#endif //SELFORGANISINGMAP_H
