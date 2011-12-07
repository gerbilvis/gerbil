#include "graphsom.h"

GraphSOM::GraphSOM(const vole::EdgeDetectionConfig &conf, int dimension)
    : SOM(conf, dimension), graph(NULL)
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

	graph->initDijkstra();

	graph->precomputePaths(false); // TODO: not sure this is right position
}

void GraphSOM::initGraph(const msi::Configuration &conf)
{

	assert((conf.graph_type == "MESH") || (conf.graph_type == "MESH_P"));

	if (conf.graph_type == "MESH")
  {
		if(conf.beta > 0.0 || conf.phi > 0.0)
		{
			std::cout << "# Using SMALL WORLD MESH topology" <<std::endl;
			graph = new msi::SW_Mesh(conf, conf.nodes);
		}
		else
		{
			std::cout << "# Using MESH topology" <<std::endl;
			graph = new msi::Mesh(conf, conf.nodes);
		}
  }
  else if (conf.graph_type == "MESH_P")
  {
		// conf.periodic=true; TODO: This does not work. graph will not be created as desired?
		// maybe make periodic an argument to the mesh, and/or remove conf.periodic altogether?
		if(conf.beta > 0.0 || conf.phi > 0.0)
		{
			std::cout << "# Using PERIODIC SMALL WORLD MESH topology" <<std::endl;
			graph = new msi::SW_Mesh(conf, conf.nodes);
		}
		else
		{
			std::cout << "# Using PERIODIC MESH topology" <<std::endl;
			graph = new msi::Mesh(conf, conf.nodes);
		}
	}
}

void GraphSOM::updateGraph()
{
	if (!graph)
		return;

	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++ ) {
			//update node information in graph
			unsigned int index = (y * (width)+x);
			Neuron* n = &neurons[y][x];
			graph->update(index, n);
		}
	}
}

cv::Point GraphSOM::identifyWinnerNeuron(const multi_img::Pixel &inputVec) const
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
			const msi::Node* node = graph->getNode((unsigned int)(y* (width) + x));
       		const Neuron *neuron = node->getNeuron();
       		dist = distfun->getSimilarity(*neuron, inputVec);
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
	unsigned int index = ((unsigned int)pos.y * width + (unsigned int)pos.x);
	const msi::Node *centerNode = graph->getNode(index);

	graph->updateNeighborhood(centerNode, radius, learnRate, input);

	graph->nextIter();
}

double GraphSOM::getDistance(const cv::Point2d &p1, const cv::Point2d &p2) const
{
	bool periodic;
	unsigned int index1;
	unsigned int index2;
	double distance;

	if(config.graph_type == "MESH_P")
		periodic = true;
	else
		periodic = false;

	cv::Point2d cp1;
	cv::Point2d cp2;

	double intpart = 0.0;
	if(std::modf(cp1.x,&intpart) >=0.5)
		cp1.x = std::ceil(p1.x);
	else
		cp1.x = std::floor(p1.x);

	intpart = 0.0;
	if(std::modf(p1.y,&intpart) >=0.5)
		cp1.y = std::ceil(p1.y);
	else
		cp1.y = std::floor(p1.y);

	intpart = 0.0;
	if(std::modf(p2.x,&intpart) >=0.5)
		cp2.x = std::ceil(p2.x);
	else
		cp2.x = std::floor(p2.x);

	intpart = 0.0;
	if(std::modf(p2.y,&intpart) >=0.5)
		cp2.y = std::ceil(p2.y);
	else
		cp2.y = std::floor(p2.y);

	index1 = (unsigned int)cp1.y * width + (unsigned int)cp1.x;
	index2 = (unsigned int)cp2.y * width + (unsigned int)cp2.x;

	distance = 0.0;

	if(config.graph_withGraph == false && config.withUMap == true)
	{
		assert(umap);

		distance = graph->getDistance(index1,index2,true);
	}

	else if(config.graph_withGraph == true && config.withUMap == true)
	{
		distance = graph->getDistance(index1,index2, true);
	}

	else if(config.graph_withGraph == true && config.withUMap == false)
	{
		if(periodic) {
			if(config.forceDD) {
				double dd,wd;
				dd = SOM::getDistance(cp1,cp2);
				wd = wrapAroundDistance(cp1,cp2);

				distance = std::min(dd, wd);
			} else {
				distance = graph->getDistance(index1,index2,false);
			}
		} else {
			distance = graph->getDistance(index1,index2,false);
		}
	}

	//actually, should never be reached
	if(distance == DBL_MAX)
	{
		std::cout <<" infinite distance!! " << index1 << " " << index2 << " " <<distance<<std::endl;
		return 1.0;
	}

	return distance;
}

double GraphSOM::wrapAroundDistance(const cv::Point2d &p1, const cv::Point2d &p2) const
{
	cv::Point2d intersection;
	cv::Point2d inter1, inter2;

	int bordersFound = 0;
	double h = (double)height;
	double w = (double)width;

	cv::Mat1d topEdge 		= cv::Mat::zeros(2,2,CV_64F);
	topEdge[0][0] = 0.0;
	topEdge[1][0] = h;
	topEdge[0][1] = 1.0;
	topEdge[1][1] = 0.0;

	cv::Mat1d leftEdge 		= cv::Mat::zeros(2,2,CV_64F);
	leftEdge[0][0] = 0.0;
	leftEdge[1][0] = 0.0;
	leftEdge[0][1] = 0.0;
	leftEdge[1][1] = 1.0;

	cv::Mat1d rightEdge 		= cv::Mat::zeros(2,2,CV_64F);
	rightEdge[0][0] = w;
	rightEdge[1][0] = 0.0;
	rightEdge[0][1] = 0.0;
	rightEdge[1][1] = 1.0;

	cv::Mat1d bottomEdge 		= cv::Mat::zeros(2,2,CV_64F);
	bottomEdge[0][0] = 0.0;
	bottomEdge[1][0] = 0.0;
	bottomEdge[0][1] = 1.0;
	bottomEdge[1][1] = 0.0;

	if(findBorderIntersection(p1, p2, intersection, topEdge))
	{
		if((intersection.x <= w) && (intersection.x >= 0.0) && (intersection.y <= h) && (intersection.y >= 0.0) )
		{
			bordersFound++;
			inter1 = intersection;
		}
	}
	if(findBorderIntersection(p1, p2, intersection, leftEdge))
	{
		if((intersection.x <= w) && (intersection.x >= 0.0) && (intersection.y <= h) && (intersection.y >= 0.0) )
		{

			if((bordersFound == 1) && (inter1 != intersection))
				inter2 = intersection;
			else
				inter1 = intersection;
			bordersFound++;
		}
	}
	if(findBorderIntersection(p1, p2, intersection, rightEdge))
	{
		if((intersection.x <= w) && (intersection.x >= 0.0) && (intersection.y <= h) && (intersection.y >= 0.0) )
		{

			if((bordersFound == 1) && (inter1 != intersection))
				inter2 = intersection;
			else
				inter1 = intersection;
			bordersFound++;
		}
	}
	if(findBorderIntersection(p1, p2, intersection, bottomEdge))
	{
		if((intersection.x <= w) && (intersection.x >= 0.0) && (intersection.y <= h) && (intersection.y >= 0.0) )
		{

			if((bordersFound == 1) && (inter1 != intersection))
				inter2 = intersection;
			else
				inter1 = intersection;
			bordersFound++;
		}
	}
	//some error occured, use direct distance, maybe not the shortest but won't produce error!
	if(bordersFound != 2)
		return SOM::getDistance(p1,p2);
	else
	{
		assert(0 == 1); // this stuff is easy... but not right now
		/*if(config.sw_initialDegree == 4)
		{
			return (manhattanDistance(inter1,inter2) - manhattanDistance(p1,p2));
		}
		else
			return (euclideanDistance(inter1,inter2) - euclideanDistance(p1,p2));*/
	}
}

bool GraphSOM::findBorderIntersection(const cv::Point2d &p1, const cv::Point2d &p2,
                                      cv::Point2d &intersect, cv::Mat1d &border) const
{

	// try to find the borders of the SOM (left,right,top,bottom) that intersect with the line,
	// defined by p1,p2 using the parameter form for lines (x_o + lambda * (p2-p1) )
	//and return the intersection point

	cv::Point2d direction = p2 - p1;

	bool success = false;
	cv::Mat solution;

	cv::Mat1d lefthandside = cv::Mat::zeros(2,2,CV_64F);
	cv::Mat1d righthandside = cv::Mat::zeros(2,1,CV_64F);

	//"fuss"punkt of line to test
	lefthandside[0][0] = direction.x;
	lefthandside[1][0] = direction.y;
	// direction vectorDistance
	lefthandside[0][1] = - border[0][1];
	lefthandside[1][1] = - border[1][1];
	//test top
	righthandside[0][0] = border[0][0] - p1.x;
	righthandside[1][0] = border[1][0] - p1.y;

	success = cv::solve(lefthandside, righthandside, solution, cv::DECOMP_LU);

	if(success)
	{
		double *solPtr = solution.ptr<double>(0);
		intersect.x = p1.x + (solPtr[0] * (p2.x - p1.x) );
		intersect.y = p1.y + (solPtr[0] * (p2.y - p1.y) );
	}

	return success;
}
