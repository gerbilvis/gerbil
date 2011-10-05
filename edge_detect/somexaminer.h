#ifndef SOMEXAMINER_H
#define SOMEXAMINER_H

class SOMExaminer
{
public:
    SOMExaminer();

	/**
	* Helper function to generate specific output names
	* consisting of the current parameter set.
	*/
	std::string getFilenameExtension();

	/**
	* RGB visualization of SOM network
	* using CIE X Y Z and D65 illumination.
	*
	* @param write write RGB-Image to disk
	*/
	void displaySom(bool write = true);

  /**
  * Node representation of graph in dot file
  *
  * @param write write converted cv::Mat to disk
  */
  void displayGraph(bool write = true);

  /**
  * Counts number of each neuron being the BMU (winning neuron)
  *
  * @param write write grayscale coded image to disk
  */
  void displayBmuMap(bool write = true);

  /**
  * Distances between SOM nodes
  *
  * @param write write image to disk
  */
  void displayDistanceMap(bool write = true);

	/**
	* Multispectral analysis tool that visualizes both the multispectral
	* image and SOM data as well as the resulting rank image.
	* Very useful to examine the conjunction between the SOM learning of the
	* multispectral input and the mapping to the rank image.
	*/
	void compareMultispectralData();

	/**
	*	\brief Visualize distances between two nodes
	*
	*	Left-click 	: Select first node
	*	Right-click 	: Select second node
	* If two nodes are selected, the computed shortest paths are visualized graphically, and the distances
	*	are displayed on console output
	*
	*/
	void displayGraphDistances();

};

#endif // SOMEXAMINER_H
