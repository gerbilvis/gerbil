#ifndef GRAPHSOM_H
#define GRAPHSOM_H

#include "self_organizing_map.h"

//graph includes
#include "Graph/smallWorldGraph.h"
#include "Graph/completingGraph.h"
#include "Graph/graph.h"
#include "Graph/misc.h"
#include "Graph/solution.h"

class GraphSOM : public SOM
{

public:
    GraphSOM(const EdgeDetectionConfig &conf, int dimension);

	~GraphSOM() {}

    void initGraph(const msi::Configuration& conf);

	void updateGraph();

	cv::Point identifyWinnerNeuron(const multi_img::Pixel &input) const;

	void updateNeighborhood(const cv::Point &pos, const multi_img::Pixel &input,
	                        double radius, double learnRate);

	inline msi::Mesh* getGraph()
	{ return msi_graph; }

private:
	msi::Mesh *msi_graph;
};

#endif // GRAPHSOM_H
