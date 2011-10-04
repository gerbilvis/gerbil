//	-------------------------------------------------------------------------------------------------------------	 	//
// 														Variants of Self-Organizing Maps																											//
// Studienarbeit in Computer Vision at the Chair of Patter Recognition Friedrich-Alexander Universitaet Erlangen		//
// Start:	15.11.2010																																																//
// End	:	16.05.2011																																																//
// 																																																									//
// Ralph Muessig																																																		//
// ralph.muessig@e-technik.stud.uni-erlangen.de																																			//
// Informations- und Kommunikationstechnik																																					//
//	---------------------------------------------------------------------------------------------------------------	//



// std library
#include <iostream>
#include <sstream>
#include <algorithm> 
#include <iterator>
#include <fstream> 

#include "self_organizing_map.h"

void SelfOrganizingMap::init() {

	if(m_graph || m_umap)
	{
    //graphSettings
		msi::Configuration graphConf;
		graphConf.output = m_conf->output_dir;
		graphConf.directed = false;
		graphConf.completing = false;
		graphConf.periodic = false;
		graphConf.insertion = msi::UNDEFINED;
		graphConf.nodes =(unsigned int) m_width*m_height;
		graphConf.startcomp = 0;
		graphConf.finishcomp = graphConf.nodes;
		graphConf.initialDegree = m_conf->sw_initialDegree;
		graphConf.graph_type = m_conf->graph_type;
		graphConf.width = m_width;
		graphConf.height = m_height;
		graphConf.sw_model = m_conf->sw_model;
		graphConf.beta = m_conf->sw_beta;
		graphConf.phi = m_conf->sw_phi;
		graphConf.maxIter = m_conf->som_maxIter;
    
		initCompletingGraph(graphConf);  
  }  
  
	// create 2D pointer array at heap of size[m_height][m_width]
	m_neurons = new Neuron**[m_height];
	for(int y = 0; y < m_height; y++) 
	{
		m_neurons[y] = new Neuron*[m_width];
		for(int x = 0; x < m_width; x++ ) 
		{
			// create neuron also on heap
			m_neurons[y][x] = new Neuron(m_vector_dim);
   
			if(m_graph || m_umap)
			{  
        //update node information
				Neuron*n = m_neurons[y][x];
#ifdef Gerbil_common        
				multi_img::Pixel p(*n);
				n->setSRGB(m_msi->bgr(p));
#endif        
				msi_graph->update((unsigned int)(y*(m_width)+x), n);
			}    
		}
	}
	if(m_graph || m_umap)
	{  
		msi_graph->initDijkstra();
  }  
}

void SelfOrganizingMap::initCompletingGraph(msi::Configuration &conf)
{

	assert( (conf.graph_type == "MESH") || (conf.graph_type == "MESH_P"));

	if(conf.graph_type == "MESH")
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
  else if(conf.graph_type == "MESH_P")
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

void SelfOrganizingMap::reallocate(unsigned int bands) 
{
	m_vector_dim = bands;
	multi_img::Value l = 0.0;
	multi_img::Value h = 1.0;
	cv::RNG rng;
	
	if(m_conf->fixedSeed)
		rng = cv::RNG (19.0);
	else
		rng = cv::RNG(cv::getTickCount());
	
	for(int y = 0; y < m_height; y++) 
	{
		for(int x = 0; x < m_width; x++ ) 
		{
			// create neuron with new size
			m_neurons[y][x] = new Neuron(bands);
			m_neurons[y][x]->randomize(rng, l, h);
		}
	}
}

void SelfOrganizingMap::randomizeNeurons(double inputDataLow, double inputDataHigh) {

	cv::RNG rng;
	if(m_conf->fixedSeed)
		rng = cv::RNG (19.0);
	else
		rng = cv::RNG(cv::getTickCount());
	
	for(int y = 0; y < m_height; y++) 
	{
		for(int x = 0; x < m_width; x++) 
		{
			m_neurons[y][x]->randomize(rng,inputDataLow, inputDataHigh);
			if(m_graph)
			{
				unsigned int index = (y * (m_width)+x);
				Neuron* n = m_neurons[y][x] ;
#ifdef Gerbil_common        
				multi_img::Pixel p(*n);
				n->setSRGB(m_msi->bgr(p));
#endif      
				msi_graph->update(index , n);
      }  
		}
	}
}

void SelfOrganizingMap::generateLookupTable(const multi_img &msi)
{
	// create 2D pointer array at heap of size[m_height][m_width]
	m_lookupTable = new cv::Point*[msi.height];
	for (int y = 0; y < msi.height; y++) {
		m_lookupTable[y] = new cv::Point[msi.width];
		for (int x = 0; x < msi.width; x++) {
			m_lookupTable[y][x] = identifyWinnerNeuron(msi(y, x));
		}
	}
}

cv::Point SelfOrganizingMap::identifyWinnerNeuron(int y, int x)
{
	return m_lookupTable[y][x];
}

    
cv::Point SelfOrganizingMap::identifyWinnerNeuron(const multi_img::Pixel &inputVec)
{
	// initialize with maximum value
	double closestDistance = DBL_MAX;
	double dist;
	// init grid position
	cv::Point winner(-1, -1);

	// find closest Neuron to inputVec in the SOM/graph
	// iterate over all neurons in grid/graph
	for (int y = 0; y < m_height; y++) 
	{
		for (int x = 0; x < m_width; x++) 
		{
			if(m_graph)
			{
				const msi::Node* node = msi_graph->getNode((unsigned int)(y* (m_width) + x));
				Neuron *neuron = node->getNeuron();
				dist = neuron->euclideanDistance(inputVec);
			} 
			else 
			{  
				dist = m_neurons[y][x]->euclideanDistance(inputVec);
			}
			// compare current distance with minimal found distance
			if (dist < closestDistance) 
			{
				// set new minimal distance and winner position
				closestDistance = dist;
				winner = cv::Point(x, y);
			}
		}
	}
	assert(winner.x >= 0);
	return winner;
}

int SelfOrganizingMap::save(std::string output, std::string params) {

	// open stream to write at
	std::ofstream file_stream(output.c_str());
	if (file_stream.bad()) {
		std::cerr << "SOM: File cannot be opened! Saving SOM not possible!" << std::endl; 
		return 1; 
	}

	// first line contains: all params of the som
	// m_heightSom, m_widthSom, bands, m_radiusStart, m_radiusEnd, m_iter, m_learningStart, m_learningEnd
	file_stream << params << std::endl;

    // datensatz in datei kopieren      
	for(int y = 0; y < m_height; y++) {
		for(int x = 0; x < m_width; x++) {
			std::copy(m_neurons[y][x]->begin(), m_neurons[y][x]->end(), std::ostream_iterator<double>(file_stream, " "));
			file_stream << std::endl;
		}
	}
	file_stream.close();

	return 0;
}

std::string SelfOrganizingMap::load(std::string input) {

	// datei oeffnen
	std::ifstream file_stream(input.c_str(), std::ifstream::in);
    if (file_stream.bad()) {
		std::cerr << "SOM: Opening file for restoring som data failed!" << std::endl; 
		return m_params; 
	}
	// read the important som parameters at the first line: height width dimension
	std::getline(file_stream, m_params);

	std::stringstream par(m_params);
	par >> m_width;
	par >> m_height;
	par >> m_vector_dim;
  
	// create 2D pointer array at heap of size[m_height][m_width]
	m_neurons = new Neuron**[m_height];
	for(int y = 0; y < m_height; y++) {

		m_neurons[y] = new Neuron*[m_width];
		for(int x = 0; x < m_width; x++ ) {
			// create neuron also on heap
			m_neurons[y][x] = new Neuron(m_vector_dim);
			// just read in every vector value
			for(int i = 0; i < m_vector_dim; i++)
				file_stream >> (*m_neurons[y][x])[i];
		}
	}
	file_stream.close();

	return m_params;
}
