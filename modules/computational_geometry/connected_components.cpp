#include <iostream>
#include <fstream>

// #include "CDebugUtils.h"
#include "connected_components.h"

namespace vole {

	ConnectedComponents::ConnectedComponents()
	{
		this->threshold = INT_MIN;
	}

	ConnectedComponents::ConnectedComponents(int threshold, cv::Mat_<int> baseArray)
		: threshold(threshold), baseArray(baseArray)
	{
	}

	ConnectedComponents::ConnectedComponents(std::vector<cv::Point_<int> > points) {
		int maxX = INT_MIN; int maxY = INT_MIN;
		for (unsigned int i = 0; i < points.size(); ++i) {
			if (points[i].x > maxX) maxX = points[i].x;
			if (points[i].y > maxY) maxY = points[i].y;
		}
		baseArray = cv::Mat_<int>(maxY+1, maxX+1, static_cast<int>(2));
		threshold = 1;
		for (unsigned int i = 0; i < points.size(); ++i) {
			baseArray[points[i].y][points[i].x] = 0;
		}
	}

void ConnectedComponents::findLargestConnectedComponent(int &number, int &size)
{
	size = INT_MIN;
	number = -1;
	for (int i = 0; i < static_cast<int>(components.size()); i++) {
		if (getSizeOfComponent(i) > size) {
			size = getSizeOfComponent(i);
			number = i;
		}
	}
}


int ConnectedComponents::computeConnectedComponents()
{
	if (baseArray.rows == 0) {
			std::cerr << "baseArray not initialized, call setArray() before calling getConnectedComponents; aborted." << std::endl;
			return 0;
	}
	if (threshold == INT_MIN) {
		std::cerr << "threshold not initialized, call setThreshold() before calling getConnectedComponents; aborted." << std::endl;
		return 0;
	}
	cv::Mat_<int> componentsArray(baseArray.rows, baseArray.cols, static_cast<int>(0));

	// ignore the very border; this simplifies bound checking and should be irrelevant for the DIS signal
	int xMin = 1; 
	int yMin = 1;
	int xMax = baseArray.cols - 1;
	int yMax = baseArray.rows - 1;

	nComponents = 0;
	equivalenceClasses.clear();
	components.clear();

	cv::Mat_<int> todo(2, componentsArray.cols * componentsArray.rows);
	for (int x = xMin; x < xMax; x++) {
		for (int y = yMin; y < yMax; y++) {
			// if the value is within the tape suspicion area and the current pixel is not part of another partition...
			if ((baseArray[y][x] >= threshold) && (componentsArray[y][x] == 0)) {
				// ...floodfill the current partition
				nComponents++;
				int todoStart = 0;
				int todoEnd = 1;
				todo[0][0] = x;
				todo[1][0] = y;
				componentsArray[y][x] = nComponents;
				while(todoStart != todoEnd) {
					int tx = todo[0][todoStart];
					int ty = todo[1][todoStart];
					if ((tx < xMax) && (componentsArray[ty][tx+1] == 0) && (baseArray[ty][tx+1] <= threshold)) {
						todo[0][todoEnd] = tx+1; todo[1][todoEnd] = ty; todoEnd++;
						componentsArray[ty][tx+1] = nComponents;
					}
					if ((tx > xMin) && (componentsArray[ty][tx-1] == 0) && (baseArray[ty][tx-1] <= threshold)) {
						todo[0][todoEnd] = tx-1; todo[1][todoEnd] = ty; todoEnd++;
						componentsArray[ty][tx-1] = nComponents;
					}
					if ((ty < yMax) && (componentsArray[ty+1][tx] == 0) && (baseArray[ty+1][tx] <= threshold)) {
						todo[0][todoEnd] = tx; todo[1][todoEnd] = ty+1; todoEnd++;
						componentsArray[ty+1][tx] = nComponents;
					}
					if ((ty > yMin) && (componentsArray[ty-1][tx] == 0) && (baseArray[ty-1][tx] <= threshold)) {
						todo[0][todoEnd] = tx; todo[1][todoEnd] = ty-1; todoEnd++;
						componentsArray[ty-1][tx] = nComponents;
					}
					todoStart++;
				}
			}
		}
	}

//	CDebugUtils::dumpCDOIntegerArray(&componentsArray, "C:\\Temp\\componentsArray.txt");
	// convert componentsArray to vector components
	for (int i = 0; i < nComponents; i++) {
		std::vector<cv::Vec2f> vec;
		components.push_back(vec);
	}
	for (int y = 0; y < yMax; y++) {
		for (int x = 0; x < xMax; x++) {
			int c = componentsArray[y][x] - 1;
			if (c > -1) components[c].push_back(cv::Vec2f(static_cast<float>(x), static_cast<float>(y)));
		}
	}
	return nComponents;
}

void ConnectedComponents::dumpComponents(std::string filename) {
	std::ofstream outFile;
	outFile.open(filename.c_str());
	for (int i = 0; i < nComponents; i++) {
		for (unsigned int j = 0; j < components[i].size(); j++) {
			outFile << "(" << components[i][j][0] << "," << components[i][j][1] << ") ";
		}
		outFile << std::endl;
	}
	outFile.close();
}

}
