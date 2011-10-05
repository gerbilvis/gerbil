#include "graphsom.h"

void GraphSOM::GraphSOM(const EdgeDetectionConfig &conf, int dimension)
    : SOM(conf, dimension), m_graph(conf.graph_withGraph),
      m_umap(conf.withUMap), msi_graph(NULL)
{
	msi::Configuration graphConf;
	graphConf.output = config.output_dir;
	graphConf.directed = false;
	graphConf.completing = false;
	graphConf.periodic = false;
	graphConf.insertion = msi::UNDEFINED;
	graphConf.nodes =(unsigned int)(width * height);
	graphConf.startcomp = 0;
	graphConf.finishcomp = graphConf.nodes;
	graphConf.initialDegree = config.sw_initialDegree;
	graphConf.graph_type = config.graph_type;
	graphConf.width = width;
	graphConf.height = height;
	graphConf.sw_model = config.sw_model;
	graphConf.beta = config.sw_beta;
	graphConf.phi = config.sw_phi;
	graphConf.maxIter = config.som_maxIter;

	initGraph(graphConf);

	updateGraph();

	msi_graph->initDijkstra();
}

void GraphSOM::initGraph(const msi::Configuration &conf)
{

	assert( (conf.graph_type == "MESH") || (conf.graph_type == "MESH_P"));

	if (conf.graph_type == "MESH")
  {
		if(conf.beta > 0.0 || conf.phi > 0.0)
		{
			std::cout << "# Using SMALL WORLD MESH topology" <<std::endl;
			msi_graph = new msi::SW_Mesh(conf, conf.nodes);
		}
		else
		{
			std::cout << "# Using MESH topology" <<std::endl;
			msi_graph = new msi::Mesh(conf, conf.nodes);
		}
  }
  else if (conf.graph_type == "MESH_P")
  {
    conf.periodic=true;
		if(conf.beta > 0.0 || conf.phi > 0.0)
		{
			std::cout << "# Using PERIODIC SMALL WORLD MESH topology" <<std::endl;
			msi_graph = new msi::SW_Mesh(conf, conf.nodes);
		}
		else
		{
			std::cout << "# Using PERIODIC MESH topology" <<std::endl;
			msi_graph = new msi::Mesh(conf, conf.nodes);
		}
	}
}

void GraphSOM::updateGraph()
{
	if (!msi_graph)
		return;

	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++ ) {
			//update node information in graph
			unsigned int index = (y * (width)+x);
			Neuron* n = m_neurons[y][x];
			msi_graph->update(index, n);
		}
	}
}

void GraphSOM::identifyWinnerNeuron(const multi_img::Pixel &inputVec)
{
	// initialize with maximum value
	double closestDistance = DBL_MAX;
	double dist;
	// init grid position
	cv::Point winner(-1, -1);

	// find closest Neuron to inputVec in the SOM/graph
	// iterate over all neurons in grid/graph
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			const msi::Node* node = msi_graph->getNode((unsigned int)(y* (width) + x));
       		const Neuron *neuron = node->getNeuron();
       		dist = config.distfun->getSimilarity(*neuron, inputVec);
			// compare current distance with minimal found distance
			if (dist < closestDistance) {
				// set new minimal distance and winner position
				closestDistance = dist;
				winner = cv::Point(x, y);
			}
		}
	}
	assert(winner.x >= 0);
	return winner;
}

void GraphSOM::updateNeighborhood(const cv::Point &pos, const multi_img::Pixel &input, double radius, double learnRate)
{
  unsigned int index = ((unsigned int)pos.y * som.getWidth() + (unsigned int)pos.x);
  msi::Mesh graph = som.getGraph();
	const msi::Node *centerNode = graph->getNode(index);

	graph->updateNeighborhood(centerNode, radius, learnRate, input);

  graph->nextIter();
}
