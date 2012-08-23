#ifndef VOLE_CONNECTED_COMPONENTS_H_
#define VOLE_CONNECTED_COMPONENTS_H_


#include "geometry_utilities.h"
#include "morphological_operators.h"

#include <vector>
#include <iostream>
#include <ostream>
#include <algorithm>

namespace vole {

/** Computes connected components on an integer matrix
*/
class ConnectedComponents
{
public:
	// computes the connected components over all entries in baseArray that are lower or equal than threshold
	ConnectedComponents();
	ConnectedComponents(int threshold, cv::Mat_<int> baseArray);
	ConnectedComponents(std::vector<cv::Point_<int> > points);

	int computeConnectedComponents();
	void findLargestConnectedComponent(int &number, int &size);
	void dumpComponents(std::string filename);

	inline void setThreshold(int threshold) {this->threshold = threshold;}
	inline void setArray(cv::Mat_<int> baseArray) {this->baseArray = baseArray;}
	inline int getNumberOfConnectedComponents() {return nComponents;}

	inline int getSizeOfComponent(int componentNumber) { return static_cast<int>(components[componentNumber].size()); }
	inline void getConnectedComponent(int number, std::vector<cv::Vec2f> &result)
	{
		if (number > nComponents)
		{
			std::cerr << "requested component id is larger than the total number of components; aborted" << std::endl;
			return;
		}
		result = components[number];
	}

private:
	int threshold;
	int nComponents;
	cv::Mat_<int> baseArray;
	std::vector<int> equivalenceClasses;
	std::vector< std::vector<cv::Vec2f> > components;

};

}

#endif // VOLE_CONNECTED_COMPONENTS_H_

