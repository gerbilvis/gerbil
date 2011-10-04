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
#include <fstream>
// opencv
#include <cv.h>
#include <highgui.h>

#include "som_trainer.h"
#include "myCanny.h"

#include "stopwatch.h"

#ifdef WITH_GERBIL_COMMON
#include "illuminant.h"
#endif


/* mouse handler indicators */
bool mouse_left_clicked;  
bool mouse_right_clicked;
unsigned int g_node1=UINT_MAX;
unsigned int g_node2=UINT_MAX;

/// global mouse handler function controlling the leftclick in the msi window
void mouse_leftclick( int event, int x, int y, int flags, void* param ) {
	
  cv::Point *mark = (cv::Point *) param;

  switch (event) {
  case CV_EVENT_MOUSEMOVE: 
    mark->x = x;
    mark->y = y;
    break;

  case CV_EVENT_LBUTTONDOWN:
    mouse_left_clicked = true;
    break;

  case CV_EVENT_LBUTTONUP:
    mouse_left_clicked = false;
    break;
  }
}

/// global mouse handler function controlling the rightclick in the som window
void mouse_rightclick( int event, int x, int y, int flags, void* param ) {

  cv::Point *mark2 = (cv::Point *) param;
	
  switch( event ){
    case CV_EVENT_MOUSEMOVE: 
      mark2->x = x;
      mark2->y = y;
      break;

    case CV_EVENT_RBUTTONDOWN:
      mouse_right_clicked = true;
      break;
			
    case CV_EVENT_RBUTTONUP:
      mouse_right_clicked = false;
      break;			
			
    case CV_EVENT_LBUTTONDOWN:
      mouse_left_clicked = true;
      break;			
			
    case CV_EVENT_LBUTTONUP:
      mouse_left_clicked = false;
      break;			
    }
}

SomTrainer::SomTrainer(SelfOrganizingMap &map, const multi_img &image,
                     const EdgeDetectionConfig &conf, const std::string &name)
  : m_som(map), m_msi(image), config(conf), m_msiName(name),
    m_widthSom(m_som.getWidth()), m_heightSom(m_som.getHeight())
{

  // copy config values, stupid
  m_learningStart = config.som_learnStart;
  m_learningEnd = config.som_learnEnd;
  m_radiusStart = config.som_radiusStart; //m_widthSom/3.;
  m_radiusEnd = config.som_radiusEnd;
  m_iter = config.som_maxIter;
//   m_sw_radiusStart = config.sw_radiusStart;
  m_verbosity = config.verbosity;
  m_evolution = 0.;
  m_nr = 0;
  m_currIter = 0;
  m_bands = m_som.getDimension();
  // just cut out ".txt" from the name
  if (m_msiName.find(".txt")) {
    m_msiName.resize(m_msiName.size() - 4);
  }
  
  m_bmuMap = cv::Mat::zeros(m_heightSom, m_widthSom, CV_64F);
  
  m_withUMap = conf.withUMap;
  m_withGraph = m_som.withGraph();

	if (m_withGraph || m_withUMap)
  {
    msi_graph = m_som.getGraph();
    msi_graph->setMSI(&m_msi);
		msi_graph->precomputePaths(false);
		double diameter = msi_graph->getDiameter();
		double charPath = msi_graph->getcharacteristicPathLength();
		m_graphProperties << "Width: " << m_som.getWidth() << std::endl;
		m_graphProperties << "Height: " << m_som.getHeight() << std::endl;
		m_graphProperties << "Diameter: " << diameter << std::endl;
		m_graphProperties << "Clustering: " << msi_graph->graphClustering() << std::endl;
		m_graphProperties << "Characteristic path length: " << charPath << std::endl;
		std::cout << "#	Diameter: " << diameter << std::endl;

		std::cout << "#	Setting radius from " << m_radiusStart;
		// radius is automatically set to half of diameter, to avoid 'flushing' of SOM, if 
		// radius is chosen to large and to avoid 'holes' if radius is chosen to small
		m_radiusStart = diameter/2.0;
		std::cout << " to " << m_radiusStart <<std::endl;
  }
}

void SomTrainer::feedNetwork() 
{

  // matrices that hold shuffled sequences of the input for number of iterations
	std::cout << "# Start feeding"  <<std::endl;
	cv::Mat_<int> shuffledY(1, m_iter);
	cv::Mat_<int> shuffledX(1, m_iter);
  
	cv::RNG rng;
	
	if(config.fixedSeed)
		rng = cv::RNG (19.0);
	else
		rng = cv::RNG(cv::getTickCount());
	
  // generate random sequence of the input x,y range
	rng.fill( shuffledY, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(m_msi.height));
	rng.fill( shuffledX, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(m_msi.width));
  
  // start the iterative feeding process here
	cv::MatConstIterator_<int> itY = shuffledY.begin();
	cv::MatConstIterator_<int> itX = shuffledX.begin();
	
	unsigned int ten = m_iter/10;
	// avoids division by zero segfault
	if(m_iter < 10)
			ten = 10;
	int round = 1;
	if(config.verbosity > 0)
		std::cout  << "  0 %" <<std::endl;
	for (; itX != shuffledX.end(); itX++, itY++) 
	{
		// extract random pixel vector from the multispectral image
		const multi_img::Pixel & vec = m_msi(*itY, *itX);
		feedSample(vec);
		if( (m_iter > 10) && (m_currIter % ten) == 0 && config.verbosity > 0)
		{	
			std::cout << " "<< (round*10) << " %" <<std::endl;
			round++;
		}	
//		enable lines below, to write SOM states

// 		writeSom(1);
// 		writeSom(50);
// 		writeSom(100);
// 		writeSom(250);
// 		writeSom(500);
// 		writeSom(1000);
// 		writeSom(2500);
// 		writeSom(5000);
// 		writeSom(10000);
// 		writeSom(20000);
// 		writeSom(30000);
  }
  if(config.verbosity > 0)
		std::cout  << "100 %" <<std::endl;

  std::cout <<"# Feeding done" <<std::endl;

	displaySom(true);
	displayMSI(false);
	displayBmuMap(false);

	if(m_withUMap)
		umatrix(false);

	if(config.isGraphical && m_withGraph) 
	{
//	compareMultispectralData();
		displayGraphDistances();
	}	
}

void SomTrainer::feedSample(const multi_img::Pixel &input) 
{
  // find best matching neuron to given input vector
	cv::Point pos = m_som.identifyWinnerNeuron(input);
	//increase winning count of neuron
	m_bmuMap[pos.y][pos.x] += 1.0;
	// BMU and their neigborhood learns weighted from the input
	if(m_withGraph)
	{
		updateGraphNeighborhood(pos, input);
	} 
	else 
	{
		updateNeighborhood(pos, input);
	}

	Neuron diff(*m_som.getNeuron(pos.x, pos.y));
	m_evolution += diff.euclideanDistance(input);
	m_currIter++;
}

void SomTrainer::updateGraphNeighborhood(cv::Point_<int> &pos, const multi_img::Pixel &input)
{
  double radius;
  double learning;
  
  unsigned int index = ((unsigned int)pos.y * m_som.getWidth() + (unsigned int)pos.x);
	const msi::Node *centerNode = msi_graph->getNode(index);
  computeAdjustedWeights(learning, radius);
  
	msi_graph->updateNeighborhood(centerNode,radius,learning, input);

  msi_graph->nextIter();
}

void SomTrainer::computeAdjustedWeights(double &learnRate, double &radius) 
{
	learnRate = m_learningStart * powl(m_learningEnd/m_learningStart, (double)m_currIter/(double)m_iter);

	radius = m_radiusStart * powl(m_radiusEnd/m_radiusStart, (double)m_currIter/(double)m_iter);
}

void SomTrainer::updateSingle(const cv::Point &pos, const multi_img::Pixel &input, double weight)
{
  // find neuron
	Neuron *currentNeuron = m_som.getNeuron(pos.x, pos.y);

	currentNeuron->update(input, weight);
}

void SomTrainer::updateNeighborhood(const cv::Point &pos, const multi_img::Pixel &input) 
{
	double learning, radius;
	computeAdjustedWeights(learning, radius);
	double rad = (m_som.getHeight() == 1 ? radius : radius*radius);

	int minX = 0; int minY = 0;
	int maxX = m_som.getWidth() - 1;
	int maxY = m_som.getHeight() - 1;

// 	static int it = 0;
// 	if (!(++it % 1000))
//     std::cerr << "update " << it << " at " << pos.x << "." << pos.y << " with radius " << radius << std::endl;
	bool finished = false;
	int y = pos.y, x;
	while (1)  // y loop
	{	
		x = pos.x;
		while (1)  // x loop
		{		
			// squared distance(topological) between winning and current neuron
			double dist = (pos.x - x) * (pos.x - x) + (pos.y - y) * (pos.y - y);

			// calculate the time- and radius dependent weighting factor
			double weightingFactor = learning * exp(-(dist)/(2.0*rad));
			// check if change is still relevant
			if (weightingFactor < 0.01) 
			{
				if (x == pos.x) // we are finished in y-direction
					finished = true;
				break;  // at least we are always finished in x-direction here
			}

			if (x <= maxX && y <= maxY)
				updateSingle(cv::Point(x, y), input, weightingFactor);
      // now with opposite indices
			int yy = pos.y+pos.y - y;
			int xx = pos.x+pos.x - x;
			if (x != xx && xx >= minX && y <= maxY)
				updateSingle(cv::Point(xx, y), input, weightingFactor);
			if (y != yy && x <= maxX && yy >= minY)
				updateSingle(cv::Point(x, yy), input, weightingFactor);
			if (x != xx && y != yy && xx >= minX && yy >= minY)
				updateSingle(cv::Point(xx, yy), input, weightingFactor);

			x++;
    }
		if (finished)
			break;
		y++;
	}
}

double SomTrainer::offlineTraining(int leaveOut) {

  std::ifstream list( (config.msi_name).c_str(), std::ifstream::in);
  if(!list.is_open()) {
    std::cerr << "Couldn't find list of multispectral images!" << m_msiName << std::endl;
    return -1.;
  }
  
  std::vector<std::string> image_filenames;
  std::vector<multi_img::Pixel> trainingSamples;

  unsigned int maxBands = 0;

  int cIter = -1;
  while(!list.eof()) {
    cIter++;
    std::string currFile;
    list >> currFile;
    if(currFile.empty()) break;
    if(cIter == leaveOut) { std::cout << "#	Leaving out file " << currFile << " for training!" << std::endl; continue; }
    image_filenames.push_back(currFile);
  }
  int nrImages = cIter-1;
  int samplesPerImage = m_iter/nrImages;

  for (int i = 0; i < nrImages; i++) {

    multi_img cImg;
    cImg.minval = 0.;
    cImg.maxval = 1.;
    cImg.read_image(config.input_dir+image_filenames[i]);
    cImg.rebuildPixels(false);

    if (cImg.size() > maxBands) maxBands = cImg.size();
    cv::Mat_<cv::Vec2i> rand(1, samplesPerImage);

		cv::RNG rng;
		
		if(config.fixedSeed)
			rng = cv::RNG (19.0);
		else
			rng = cv::RNG(cv::getTickCount());
		
    rng.fill(rand, cv::RNG::UNIFORM, cv::Scalar(0,0), cv::Scalar(cImg.width, cImg.height));
    
    for (int i = 0; i < samplesPerImage; i++) {
      cv::Point pos(rand.at<cv::Vec2i>( 0, i));
      trainingSamples.push_back(cImg(pos.y, pos.x));
    }
  }
  std::cout << "#	Training samples successfully read!" << std::endl;
  
  if( maxBands > m_som.getDimension() ) {
    m_som.reallocate(maxBands);
    m_bands = maxBands;
  }

  // randomize order of training samples
  std::random_shuffle(trainingSamples.begin(), trainingSamples.end());

  std::cout << "#	Feeding the SOM network..." << std::endl;
  for(unsigned int i = 0; i < trainingSamples.size(); i++) {
    feedSample(trainingSamples[i]);
  }

  std::cout << "#	Training finished. Saving SOM..." << std::endl;
  std::stringstream s;
  s << leaveOut;

  double balance = generateBWSom();
  std::cout << "#	Balancing of the SOM: " << balance << std::endl;

  // write the visualization of SOM
  if(config.verbosity > 2 ) {
    displaySom(true);
    cv::imwrite(config.output_dir+getFilenameExtension() + "_" + s.str() + "_bwsom.png", m_img_bwsom);
  }

  m_som.save(config.output_dir+getFilenameExtension() + "_" + s.str() + ".som", getParams() );

  return balance;
}


void SomTrainer::displayGraphDistances()
{

	displaySom(true);

  mouse_left_clicked = false;

  cv::Point mark(0,0);
  cv::Point mark2(0,0);

  double scaleX = (double)(1.0*m_msi.height) / (m_widthSom);
  double scaleY = (double)(1.0*m_msi.height) / (m_heightSom);
	
	cv::Mat3f som;
  cv::Mat3f somCopy;

	cv::resize(m_img_som, som, cv::Size(256,256), 3.0, 3.0, cv::INTER_NEAREST);

  som.copyTo(somCopy);

  cv::namedWindow("SOM", CV_GUI_EXPANDED);

  cvSetMouseCallback( "SOM", mouse_rightclick, (void*)&mark2);

	cv::imshow("SOM", somCopy);

	bool display = true;
	
	while(display) 
	{

		if(mouse_right_clicked) 
		{
			cv::Point somPoint(static_cast<int>(mark2.x/scaleX) , static_cast<int>(mark2.y/scaleY));
			if(somPoint.x < m_widthSom && somPoint.y < m_heightSom) 
			{
				std::stringstream ss;
				unsigned int index = (somPoint.y * m_som.getWidth() + somPoint.x);
				g_node2 = index;
				const msi::Node* node = msi_graph->getNode(index);
				const std::vector<msi::Node*>& edg = node->getEdges();

				ss << "(" << somPoint.y << "," << somPoint.x << ") index: " << index;
				
				for(unsigned int n = 0; n < edg.size(); n++)
				{
					int col, row;
					col =  (edg.at(n))->getIndex() % m_som.getWidth();
					row =  (edg.at(n))->getIndex() / m_som.getWidth();
					cv::Point start(static_cast<int>(col * scaleX), static_cast<int>(row * scaleY));
					cv::Point end(static_cast<int>((col + 1) * scaleX), static_cast<int>((row + 1) * scaleY));
					
// 					if(edg.size()== 11)cv::rectangle( som, start, end, cv::Scalar(0,255,255), 2 );
// 					if(edg.size()== 10)cv::rectangle( som, start, end, cv::Scalar(0,255,196), 2 );
// 					if(edg.size()== 9)cv::rectangle( som, start, end, cv::Scalar(0,255,96), 2 );
// 					if(edg.size()== 8)cv::rectangle( som, start, end, cv::Scalar(0,255,0), 2 );
// 					if(edg.size()== 7)cv::rectangle( som, start, end, cv::Scalar(255,196,0), 2 );
// 					if(edg.size()== 6)cv::rectangle( som, start, end, cv::Scalar(255,155,0), 2 );
// 					if(edg.size()== 5)cv::rectangle( som, start, end, cv::Scalar(255,128,0), 2 );
// 					if(edg.size()== 4)cv::rectangle( som, start, end, cv::Scalar(255,108,0), 2 );
// 					if(edg.size()== 3)cv::rectangle( som, start, end, cv::Scalar(255,78,0), 2 );
// 					if(edg.size()== 2)cv::rectangle( som, start, end, cv::Scalar(255,55,0), 2 );
// 					if(edg.size()== 1)cv::rectangle( som, start, end, cv::Scalar(255,28,0), 2 );

				}	
				
				cv::Point start(static_cast<int>(somPoint.x * scaleX), static_cast<int>(somPoint.y * scaleY));
				cv::Point end(static_cast<int>((somPoint.x + 1) * scaleX), static_cast<int>((somPoint.y + 1) * scaleY));
				std::cout << start.x << " " << start.y << "   " <<end.x << " " <<end.y << std::endl;
				cv::rectangle( som, start, end, cv::Scalar(0,0,255), 2 );
				cv::putText( som, ss.str(), cv::Point(static_cast<int>((somPoint.x + 0.2)*scaleX),static_cast<int>((somPoint.y + 0.5)*scaleY)), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(1., 1., 1.), 2, 8, false);
										
				cv::imshow("SOM", som);
				som = somCopy.clone();
				
				if(g_node1 != UINT_MAX && g_node2 != UINT_MAX)
				{
					std::cout << "Distance " << g_node1 << " -- " << g_node2 << " : " << msi_graph->getDistance(g_node1,g_node2,false,true) <<std::endl;
					if(m_withUMap)
						std::cout << "Weighted distance " << g_node1 << " -- " << g_node2 << " : " << msi_graph->getDistance(g_node1,g_node2,true,true)<<std::endl;;
					std::vector<unsigned int> path = msi_graph->getPath(g_node1,g_node2);
					
					for(unsigned int n = 0; n < path.size();n++)
					{
						int col, row;
						col =  path.at(n) % m_som.getWidth();
						row =  path.at(n) / m_som.getWidth();
						cv::Point start(static_cast<int>(col * scaleX), static_cast<int>(row * scaleY));
						cv::Point end(static_cast<int>((col + 1) * scaleX), static_cast<int>((row + 1) * scaleY));
						
						cv::rectangle( som, start, end, cv::Scalar(255,255,0), 2 );
					}
					if(m_withUMap)
					{	
						std::vector<unsigned int> weightedPath = msi_graph->getWeightedPath(g_node1,g_node2);
						for(unsigned int n = 0; n < weightedPath.size();n++)
						{
							int col, row;
							col =  weightedPath.at(n) % m_som.getWidth();
							row =  weightedPath.at(n) / m_som.getWidth();
							cv::Point start(static_cast<int>(col * scaleX), static_cast<int>(row * scaleY));
							cv::Point end(static_cast<int>((col + 1) * scaleX), static_cast<int>((row + 1) * scaleY));
							
							cv::rectangle( som, start, end, cv::Scalar(255,0,255), 2 );
						}		
					}						
					
					std::cout <<std::endl;
				}
			}
		}
		if(mouse_left_clicked) 
		{
			cv::Point somPoint(static_cast<int>(mark2.x/scaleX) , static_cast<int>(mark2.y/scaleY));
			if(somPoint.x < m_widthSom && somPoint.y < m_heightSom) 
			{
				std::stringstream ss;
				unsigned int index = (somPoint.y * m_som.getWidth() + somPoint.x);
				g_node1 = index;
				const msi::Node* node = msi_graph->getNode(index);
				const std::vector<msi::Node*>& edg = node->getEdges();

				ss << "(" << somPoint.y << "," << somPoint.x << ") index: " << index;
				
				for(unsigned int n = 0; n < edg.size(); n++)
				{
					int col, row;
					col =  (edg.at(n))->getIndex() % m_som.getWidth();
					row =  (edg.at(n))->getIndex() / m_som.getWidth();
					cv::Point start(static_cast<int>(col * scaleX), static_cast<int>(row * scaleY));
					cv::Point end(static_cast<int>((col + 1) * scaleX), static_cast<int>((row + 1) * scaleY));
					
// 					if(edg.size()== 11)cv::rectangle( som, start, end, cv::Scalar(0,255,255), 2 );
// 					if(edg.size()== 10)cv::rectangle( som, start, end, cv::Scalar(0,255,196), 2 );
// 					if(edg.size()== 9)cv::rectangle( som, start, end, cv::Scalar(0,255,96), 2 );
// 					if(edg.size()== 8)cv::rectangle( som, start, end, cv::Scalar(0,255,0), 2 );
// 					if(edg.size()== 7)cv::rectangle( som, start, end, cv::Scalar(255,196,0), 2 );
// 					if(edg.size()== 6)cv::rectangle( som, start, end, cv::Scalar(255,155,0), 2 );
// 					if(edg.size()== 5)cv::rectangle( som, start, end, cv::Scalar(255,128,0), 2 );
// 					if(edg.size()== 4)cv::rectangle( som, start, end, cv::Scalar(255,108,0), 2 );
// 					if(edg.size()== 3)cv::rectangle( som, start, end, cv::Scalar(255,78,0), 2 );
// 					if(edg.size()== 2)cv::rectangle( som, start, end, cv::Scalar(255,55,0), 2 );
// 					if(edg.size()== 1)cv::rectangle( som, start, end, cv::Scalar(255,28,0), 2 );
				}	
				
				cv::Point start(static_cast<int>(somPoint.x * scaleX), static_cast<int>(somPoint.y * scaleY));
				cv::Point end(static_cast<int>((somPoint.x + 1) * scaleX), static_cast<int>((somPoint.y + 1) * scaleY));
				cv::rectangle( som, start, end, cv::Scalar(0,0,255), 2 );
				cv::putText( som, ss.str(), cv::Point(static_cast<int>((somPoint.x + 0.2)*scaleX),static_cast<int>((somPoint.y + 0.5)*scaleY)), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(1., 1., 1.), 2, 8, false);
										
				cv::imshow("SOM", som);
				som = somCopy.clone();
				
				if(g_node1 != UINT_MAX && g_node2 != UINT_MAX)
				{
					std::cout << "Distance " << g_node1 << " -- " << g_node2 << " : " << msi_graph->getDistance(g_node1,g_node2,false,true)<<std::endl;;
					if(m_withUMap)
						std::cout << "Weighted distance " << g_node1 << " -- " << g_node2 << " : " << msi_graph->getDistance(g_node1,g_node2,true,true)<<std::endl;;
					
					if(!msi_graph->periodic())
					{
						std::vector<unsigned int> path = msi_graph->getPath(g_node1,g_node2);
						for(unsigned int n = 0; n < path.size();n++)
						{
							int col, row;
							col =  path.at(n) % m_som.getWidth();
							row =  path.at(n) / m_som.getWidth();
							cv::Point start(static_cast<int>(col * scaleX), static_cast<int>(row * scaleY));
							cv::Point end(static_cast<int>((col + 1) * scaleX), static_cast<int>((row + 1) * scaleY));
							
							cv::rectangle( som, start, end, cv::Scalar(255,255,0), 2 );
						}
						if(m_withUMap)
						{	
							std::vector<unsigned int> weightedPath = msi_graph->getWeightedPath(g_node1,g_node2);
							for(unsigned int n = 0; n < weightedPath.size();n++)
							{
								int col, row;
								col =  weightedPath.at(n) % m_som.getWidth();
								row =  weightedPath.at(n) / m_som.getWidth();
								cv::Point start(static_cast<int>(col * scaleX), static_cast<int>(row * scaleY));
								cv::Point end(static_cast<int>((col + 1) * scaleX), static_cast<int>((row + 1) * scaleY));
								
								cv::rectangle( som, start, end, cv::Scalar(255,0,255), 2 );
							}		
						}
					}
				}
			}
		}	

		cv::waitKey(60);	
	}		
}

void SomTrainer::compareMultispectralData() 
{
#ifdef WITH_GERBIL_COMMON
	displayMSI(true);
	displaySom(true);

	mouse_left_clicked = false;

	cv::Point mark(0,0);
	cv::Point mark2(0,0);

	double scaleX = (double)(m_msi.width) / (m_widthSom);
	double scaleY = (double)(m_msi.height) / (m_heightSom);

	cv::Mat msi;
	cv::Mat3f som;
	cv::Mat3f somCopy;
	cv::resize(m_img_som, som, cv::Size(m_msi.width, m_msi.height), 0., 0., cv::INTER_NEAREST);
	som.copyTo(somCopy);

	m_img_msi = m_msi.bgr();
	m_img_msi.convertTo( msi, CV_8UC3, 255.);

  
	cv::namedWindow("Multispectral Image", 1);
	cv::namedWindow("SOM", 1);

	cvSetMouseCallback( "Multispectral Image", mouse_leftclick, (void*)&mark);
	cvSetMouseCallback( "SOM", mouse_rightclick, (void*)&mark2);

	cv::imshow("Multispectral Image", msi);
	cv::imshow("SOM", som);

  if( !m_img_rank.empty() ) 
	{
		cv::imshow("Rank Image", m_img_rank);
  }

	int rank = 0;
	int posX = mark.x;
	int posY = mark.y;

	while(true) 
	{
		if(mouse_right_clicked) 
		{
			cv::Point somPoint(static_cast<int>(mark2.x/scaleX) , static_cast<int>(mark2.y/scaleY));
			if(somPoint.x < m_widthSom && somPoint.y < m_heightSom) 
			{
				if( !m_rank.empty() ) 
					rank = m_rank.at<int>(somPoint);
				else 
					rank = somPoint.x;

				std::stringstream ss;
				ss << rank;
        
				cv::Point start(static_cast<int>(somPoint.x * scaleX), static_cast<int>(somPoint.y * scaleY));
				cv::Point end(static_cast<int>((somPoint.x + 1) * scaleX), static_cast<int>((somPoint.y + 1) * scaleY));
				cv::rectangle( som, start, end, cv::Scalar(0,0,0), 2 );
				cv::putText( som, ss.str(), cv::Point(static_cast<int>((somPoint.x + 0.2)*scaleX),static_cast<int>((somPoint.y + 0.5)*scaleY)), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(1., 1., 1.), 2, 8, false);
				std::cout << rank << " " << std::endl;
        
				cv::imshow("SOM", som);
				som = somCopy.clone();
			}
		}

		if(mouse_left_clicked && mark.y < m_msi.height && mark.x < m_msi.width) 
		{
			Neuron atClick(m_msi(mark.y, mark.x));
			Neuron old_click( m_msi(posY,posX));
			std::cout << " Difference to previous clicked vector: " << old_click.euclideanDistance(atClick) << std::endl;
			posX = mark.x;
			posY = mark.y;
			cv::Point somPoint = m_som.identifyWinnerNeuron(atClick);

			if( !m_rank.empty() ) 
				rank = m_rank.at<int>(somPoint);
      else 
				rank = somPoint.x;

			std::stringstream ss;
			ss << rank;

			cv::Point start( static_cast<int>(somPoint.x * scaleX), static_cast<int>(somPoint.y * scaleY));
			cv::Point end(static_cast<int>((somPoint.x + 1) * scaleX), static_cast<int>((somPoint.y +1) * scaleY));
			cv::rectangle( som, start, end, cv::Scalar(0,0,0), 2 );
			cv::putText( som, ss.str(), cv::Point(static_cast<int>((somPoint.x + 0.2)*scaleX),static_cast<int>((somPoint.y+0.5)*scaleY)), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(1., 1., 1.), 2, 8, false);
			std::cout << "Rank: " << rank << " Difference to input: " << atClick.euclideanDistance(*m_som.getNeuron(somPoint.x,somPoint.y))  << std::endl;
      
			cv::imshow("SOM", som);
			som = somCopy.clone();
		}
    cv::waitKey(60);
  }
#endif
}

void SomTrainer::displaySom(bool write) 
{
#ifdef WITH_GERBIL_COMMON
  /* NOTE: if we do not have the source image, we do not know _anything_ about
     its filters. So we _cannot_ create a truthful RGB image.
     If som needs to be displayed without image at hand, store the meta data
     and create empty image with correct meta data.
   */
	if (m_img_som.empty())
	{
		m_img_som = cv::Mat3f(m_som.getHeight(), m_som.getWidth());
	}

	for(int y = 0; y < m_heightSom; y++) 
	{
		cv::Vec3f *row = m_img_som[y];
		for(int x = 0; x < m_widthSom; x++) 
		{
			Neuron *n = m_som.getNeuron(x,y);

			multi_img::Pixel p(*n);
			row[x] = m_msi.bgr(p);
		}
	}

	cv::Mat3f somShow;
	int height;
	if (m_heightSom == 1)
	{	
		height = 100;
		cv::resize(m_img_som, somShow, cv::Size(m_msi.width, height), 0., 0., cv::INTER_NEAREST);
	}			

	if (write) 
	{
		m_img_som.convertTo(somShow, CV_8UC3, 255.);
		std::string graph = "";
		
		if(m_withUMap)
			graph += "";

    cv::imwrite(config.output_dir+getFilenameExtension()+ graph +"_som.png", somShow);
	std::cout << "wrote SOM to " << config.output_dir+getFilenameExtension()+ graph +"_som.png" << std::endl;
  }
#endif  
}

void SomTrainer::writeSom(int iter)
{
#ifdef WITH_GERBIL_COMMON
	cv::Mat somShow;
	int height;

	if(m_currIter != iter)
		return;

	if (m_img_som.empty())
	{
		m_img_som = cv::Mat3f(m_som.getHeight(), m_som.getWidth());
	}

	for(int y = 0; y < m_heightSom; y++) 
	{
		cv::Vec3f *row = m_img_som[y];
		for(int x = 0; x < m_widthSom; x++) 
		{
			Neuron *n = m_som.getNeuron(x,y);
			multi_img::Pixel p(*n);
			row[x] = m_msi.bgr(p);
		}
	}
  
	if (m_heightSom == 1)
	{	
		cv::Mat somScaled;
		height = 100;
		cv::resize(m_img_som, somScaled, cv::Size(m_msi.width, height), 0., 0., cv::INTER_NEAREST);
		somScaled.convertTo(somShow, CV_8UC3, 255.);
	}
	else
		m_img_som.convertTo(somShow, CV_8UC3, 255.);
	
	std::stringstream iterStream;
	iterStream << iter;
	std::string graph = "";

  cv::imwrite(config.output_dir+getFilenameExtension()+ graph +"_som_" + iterStream.str() + ".png", somShow);  
#endif
}

void SomTrainer::displayMSI(bool write) 
{
#ifdef WITH_GERBIL_COMMON

	cv::Mat3f msiShow = m_msi.bgr();
 
	if (write) 
	{
		msiShow.convertTo(msiShow, CV_8UC3, 255.);
		cv::imwrite(config.output_dir+m_msiName + "_msi.png", msiShow);
	}
#endif  
}

void SomTrainer::displayGraph(bool write) {

#ifdef WITH_GERBIL_COMMON
	assert(msi_graph);

	cv::Mat3f graphSom = cv::Mat3f(m_heightSom,m_widthSom);
   
	for(int y = 0; y < m_heightSom; y++)
	{
		cv::Vec3f *vec = graphSom[y];
		for(int x = 0; x < m_widthSom; x++)
		{
			vec[x] = msi_graph->getNode(y* m_widthSom + x)->getNeuron()->getRGB();
		}  
	}
  
	cv::Mat3f graphSomShow;
  
	int height;
	if (m_heightSom == 1)
		height = 100;
	else if (m_msi.empty())
		height = 512;
	else
		height = m_msi.width;

	if (m_msi.empty())
		cv::resize(graphSom, graphSomShow, cv::Size(512, height), 0.0, 0.0, cv::INTER_NEAREST);
	else
		cv::resize(graphSom, graphSomShow, cv::Size(m_msi.width, height), 0.0, 0.0, cv::INTER_NEAREST);

	if (write) 
	{
		graphSomShow.convertTo(graphSomShow, CV_8UC3, 255.);
		std::string graph = "";
		cv::imwrite(config.output_dir+ getFilenameExtension() + graph + "_graphSOM.png", graphSomShow);
	}
#endif
}

void SomTrainer::displayBmuMap(bool write)
{
	cv::Mat_<uchar> bmuShow(m_heightSom, m_widthSom);
	double maxIntensity = 0.0;
	
	for(int y = 0; y < m_bmuMap.rows; y++)
	{
		double *rowPtr =  m_bmuMap[y];
		for(int x = 0; x < m_bmuMap.cols; x++)
		{
			//lift values logarithmic for viewing purpose
			rowPtr[x] += 1.0;
			rowPtr[x] = std::log((rowPtr[x]));
			
			if(rowPtr[x] > maxIntensity)
			{
				maxIntensity = rowPtr[x];
			}	
		}	
	}

	for(int y = 0; y < m_bmuMap.rows; y++)
	{
		double *rowPtr =  m_bmuMap[y];
		uchar * showPtr = bmuShow.ptr<uchar>(y);
		for(int x = 0; x < m_bmuMap.cols; x++)
		{
			showPtr[x] = static_cast<uchar>( 255.0 * (rowPtr[x] / maxIntensity));
		}	
	}
	
	if (write) 
	{
		cv::imwrite(config.output_dir+ getFilenameExtension() + "_bmu_map.png", bmuShow);
		//write gnuplot data
		std::ofstream out;
		std::string fn = config.output_dir+getFilenameExtension() +"_bmu_map.dat";
		out.open(fn.c_str(), std::ofstream::out | std::ofstream::trunc);
		assert(out.is_open());

		out << "# Number of winnings for neurons "<< std::endl;
		out << "# x\ty\tWinnings\t"<< std::endl;

		for(int y = 0; y < m_bmuMap.rows ; y++ )
		{
			double *uPtr = m_bmuMap[y];
					
			for(int x = 0; x < m_bmuMap.cols; x++ )
			{
				out << x <<"\t" << y <<  "\t" <<uPtr[x]<<"\t" << std::endl;
			}
			out << "\n";
		}
		out.close();		
  }	
}

// void SomTrainer::displayDistanceMap(bool write)
// {
// 	cv::Mat1d edgeShow;
// 	if(m_verbosity > 2) 
// 	{
// 		cv::resize(m_distanceMap, edgeShow, cv::Size(512, 512), 0., 0., cv::INTER_NEAREST);
// //     cv::namedWindow("Distance Map", 1);
// //     cv::imshow("Distance Map",edgeShow);                 
// //     cv::waitKey(0);
//   }
//   if (write) 
//   {
//     edgeShow.convertTo(edgeShow, CV_8UC1, 255.);
//     cv::imwrite(config.output_dir+m_msiName + "_graph_distance_map.png", edgeShow);
//   }
// 
// }

cv::Mat1d SomTrainer::umatrix(bool write)
{
	int width = m_som.getWidth();
	int height = m_som.getHeight();
		
	double maxIntensity =0.0;
	double minIntensity =DBL_MAX;
		
	Neuron *center;
	Neuron *topLeft;
	Neuron *topCenter;
	Neuron *topRight;
	Neuron *centerLeft;
	Neuron *centerRight;
	Neuron *bottomLeft;
	Neuron *bottomCenter;
	Neuron *bottomRight;
		
	//touching border?
	bool l,t,r,b;
	double cross, diagonal;
	int neighbors ;

	bool periodic;
	bool sw = false;
	int initialDegree;
	
	if(m_withGraph)
	{
		assert(config.graph_type == "MESH" || config.graph_type == "MESH_P");
		periodic = msi_graph->periodic();
		if(config.sw_beta > 0.0 || config.sw_phi > 0.0)
			sw = true;
		initialDegree = config.sw_initialDegree;
		maxIntensity = 1.0;
	}
	else
	{
		periodic = false;
		initialDegree = 4;
	}	

	umap = cv::Mat::zeros(height, width, CV_64F);
	for(int y = 0; y < height; y++)
	{
		double *uPtr = umap[y];
		for(int x = 0; x < width; x++)
		{
			neighbors = 0;
	
			t = false;
			l = false;
			r = false;
			b = false;
				
			cross = 0.0;
			diagonal = 0.0;
					
			if(y == 0 )
				t = true;
					
			if(x == 0)
				l = true;
				
			if(y == (height-1))
				b = true;
				
			if(x == (width-1))
				r = true;
				center = m_som.getNeuron(x,y);
				
			if(t == false)
			{	
				topCenter = m_som.getNeuron(x,y-1);
				multi_img::Pixel p_tc(*topCenter);
				cross += center->euclideanDistance(p_tc);
				msi_graph->setEdgeWeight((y*width +x),((y-1)*width +x), center->euclideanDistance(p_tc));
				neighbors++;
			}
			else
			{
				if(periodic)
				{
					int h = (height-1);
					topCenter = m_som.getNeuron(x,h);
					multi_img::Pixel p_tc(*topCenter);
					cross += center->euclideanDistance(p_tc);
					msi_graph->setEdgeWeight((y*width +x),((h)*width +x), center->euclideanDistance(p_tc));
					neighbors++;
				}	
			}	
				
			if(l == false)
			{	
				centerLeft = m_som.getNeuron(x-1,y);
				multi_img::Pixel p_cl(*centerLeft);
				cross += center->euclideanDistance(p_cl);
				msi_graph->setEdgeWeight((y*width +x),((y)*width +(x-1)), center->euclideanDistance(p_cl));
				neighbors++;
			}
			else
			{
				if(periodic)
				{
					int w = (width-1);
					centerLeft = m_som.getNeuron(w,y);
					multi_img::Pixel p_cl(*centerLeft);
					cross += center->euclideanDistance(p_cl);
					msi_graph->setEdgeWeight((y*width +x),((y)*width +w), center->euclideanDistance(p_cl));
					neighbors++;
				}	
			}	
				
			if(r == false)
			{	
				centerRight = m_som.getNeuron(x+1,y);
				multi_img::Pixel p_cr(*centerRight);
				cross += center->euclideanDistance(p_cr);
				msi_graph->setEdgeWeight((y*width +x),((y)*width +(x+1)), center->euclideanDistance(p_cr));
				neighbors++;
			}
			else
			{
				if(periodic)
				{
					int w = 0;
					centerRight = m_som.getNeuron(w,y);
					multi_img::Pixel p_cr(*centerRight);
					cross += center->euclideanDistance(p_cr);
					msi_graph->setEdgeWeight((y*width +x),((y)*width +w), center->euclideanDistance(p_cr));
					neighbors++;
				}	
			}	
				
			if(b == false)
			{	
				bottomCenter = m_som.getNeuron(x,y+1);
				multi_img::Pixel p_bc(*bottomCenter);
				cross += center->euclideanDistance(p_bc);
				msi_graph->setEdgeWeight((y*width +x),((y+1)*width +x), center->euclideanDistance(p_bc));
				neighbors++;
			}
			else
			{
				if(periodic)
				{
					int h = 0;
					bottomCenter = m_som.getNeuron(x,h);
					multi_img::Pixel p_bc(*bottomCenter);
					cross += center->euclideanDistance(p_bc);
					msi_graph->setEdgeWeight((y*width +x),((h)*width +x), center->euclideanDistance(p_bc));
					neighbors++;
				}	
			}	
//diagonal edges
			if((l | t) == false && (initialDegree == 8))
			{
				topLeft = m_som.getNeuron(x-1,y-1);
				multi_img::Pixel p_tl(*topLeft);

				diagonal += center->euclideanDistance(p_tl) * (1.0/std::sqrt(2.0));
				msi_graph->setEdgeWeight((y*width +x),((y-1)*width +(x-1)), center->euclideanDistance(p_tl) );
				neighbors++;
			}
			
			if((r | t) == false && (initialDegree == 8)	)
			{	
				topRight = m_som.getNeuron(x+1,y-1);
				multi_img::Pixel p_tr(*topRight);

				diagonal += center->euclideanDistance(p_tr) * (1.0/std::sqrt(2.0));
				msi_graph->setEdgeWeight((y*width +x),((y-1)*width +(x+1)), center->euclideanDistance(p_tr) );
				neighbors++;
			}
					
			if((b | l) == false && (initialDegree == 8))
			{	
				bottomLeft = m_som.getNeuron(x-1,y+1);
				multi_img::Pixel p_bl(*bottomLeft);

				diagonal += center->euclideanDistance(p_bl) * (1.0/std::sqrt(2.0));	
				msi_graph->setEdgeWeight((y*width +x),((y+1)*width +(x-1)), center->euclideanDistance(p_bl));
				neighbors++;
			}
				
			if((b | r) == false && (initialDegree == 8))
			{	
				bottomRight = m_som.getNeuron(x+1,y+1);
				multi_img::Pixel p_br(*bottomRight);

				diagonal += center->euclideanDistance(p_br) * (1.0/std::sqrt(2.0));
				msi_graph->setEdgeWeight((y*width +x),((y+1)*width +(x+1)), center->euclideanDistance(p_br));
				neighbors++;
			}	
				
			uPtr[x] = (diagonal + cross) * (1.0/ neighbors);

			if(uPtr[x] > maxIntensity)maxIntensity = uPtr[x];
			if(uPtr[x] < minIntensity)minIntensity = uPtr[x];
		}
	}
		
	if(m_withUMap)
		msi_graph->scaleDistances(config.scaleUDistance);

	if(write)
	{
		cv::Mat_<uchar> umatrix(height, width);
		
		for(int y = 0; y < height; y++)
		{
			double *uPtr = umap[y];
			uchar *writePtr = umatrix.ptr<uchar>(y);
			for(int x = 0; x < width; x++)
			{
				writePtr[x] = static_cast<uchar>(255.0 * (uPtr[x] / maxIntensity));
			}
		}
		
		std::string graph = "";
		if(m_withGraph)
		{
			graph += "_graph";
			if(config.sw_phi > 0.0 || config.sw_beta > 0.0)
				graph += "_sw_graph";	
			
		}	
		
		cv::imwrite(config.output_dir+getFilenameExtension() + graph +"_umatrix.png", umatrix);
	
		//write gnuplot data
		std::ofstream out;
		std::string fn = config.output_dir+getFilenameExtension() + graph +"_umatrix.dat";
		out.open(fn.c_str(), std::ofstream::out | std::ofstream::trunc);
		assert(out.is_open());

		out << "# Unified Distance Map for SOM "<< std::endl;
		out << "# x\ty\tU-value\t"<< std::endl;

		for(int y = 0; y < umap.rows; y++ )
		{

			double *uPtr = umap[y];
					
			for(int x = 0; x < umap.cols; x++ )
			{
				out << y <<"\t" << x <<  "\t" <<uPtr[x]<<"\t" << std::endl;
			}
			out << "\n";

		}
		out.close();		
		std::cout << "# Wrote U-Map" <<std::endl;
	}
	
	return umap;
}


double SomTrainer::generateBWSom() {

  int width = m_som.getWidth();
  int height = m_som.getHeight();

  m_img_bwsom = cv::Mat_<float>(height, width);

  int radius = 1;

  // determine radius
  if( height % 2 == 0) {
    int order = 0;
    int length = 2;
    for(order = 1; order < 10; order++) {
      length *= 2;
      if(length == height) { break; }
    }
    radius = order;//static_cast<unsigned int>(pow(2., order));
  } else if( height % 3 == 0 ) {
    int order = 0;
    int length = 3;
    for(order = 1; order < 10; order++) {
      if(length == height) { break; }
      length *= 3;
    }
    radius = order;//static_cast<unsigned int>(pow(2., order));
  }
  if(height == 1 ) {
    if( width < 16 ) {
      radius = 3;
    } else {
      radius = 1 + width/10;
    }
  }
  radius = 1;

  double totalDiff = 0.;

  for(int y = 0; y < height; y++) {
    for(int x = 0; x < width; x++) {

      double difference = 0.;
      Neuron *cn = m_som.getNeuron(x,y);
      int count = 0;

      for(int posY = y-radius; posY <= y+radius; posY++) {
        if(posY < 0 || posY >= height) { continue; }
        for(int posX = x-radius; posX <= x+radius; posX++) {
          if(posX < 0 || posX >= width) { continue; }
          //double dist = (x - posX)*(x - posX) + (y - posY)*(y - posY);
        //  if(dist <= radius) {
            count++;
            Neuron *neighbor = m_som.getNeuron(posX,posY);
            difference += cn->euclideanDistance(*neighbor);
        //  }
        }
      }
      if(count > 1) { difference = difference / static_cast<double>(count-1); }
      totalDiff += difference;
      m_img_bwsom(y,x) = static_cast<float>(difference);
    }
  }
  // normalize on number of neurons
  totalDiff /= (double)(width*height);

  return totalDiff;
}

cv::Mat SomTrainer::generateRankImage() {

	m_img_rank = cv::Mat1f(m_msi.height, m_msi.width);
	float normidx = 1.0f / (float)(m_widthSom*m_heightSom);

	bool indirect = !m_rank.empty();

	for (int y = 0; y < m_msi.height; y++) {
		float *row = m_img_rank[y];
		for (int x = 0; x < m_msi.width; x++) {
			cv::Point p = m_som.identifyWinnerNeuron(m_msi(y, x));
			if (indirect)
				row[x] = m_rank(p) * normidx;
			else
				row[x] = p.x * normidx;
		}
	}


	double min, max;
	cv::minMaxLoc(m_img_rank, &min, &max);
	std::cerr << "rank image: [" << min << ", " << max << "]" << std::endl;

	cv::imwrite(config.output_dir + getFilenameExtension()+"_rank.png", m_img_rank*255.0f);

	if(config.isGraphical) {
		compareMultispectralData();
	}

	return m_img_rank;

}

cv::Mat SomTrainer::generateRankImage(cv::Mat_<unsigned int> &rankMatrix) {
  m_rank = rankMatrix;
	cv::imwrite(config.output_dir + getFilenameExtension()+"_rankmatrix.png", m_rank);

  return generateRankImage();
}

void SomTrainer::getEdge( cv::Mat1d &dx, cv::Mat1d &dy, int mode)
{
	std::cout << "# Setting up neuron lookup table" << std::endl;
	// actually oart of the training, instead of edge image
	m_som.generateLookupTable(m_msi);
	std::cout << "# Done!" << std::endl;	
	std::cout << "# Calculating derivatives (dx, dy )" << std::endl;
	
	dx = cv::Mat::zeros(m_msi.height, m_msi.width, CV_64F);
  dy = cv::Mat::zeros(m_msi.height, m_msi.width, CV_64F);
	
	//filter coefficients
	double c1,c2,c3;
	
	cv::Point point;
	
	double distX,distY;
	distX = 0.0;
	distY = 0.0;
	
	double p1,p2;
	p1 = 0.0;
	p2 = 0.0;
	
	double fraction = 1.0;
	
	//use Sobel mask
	if(mode == 0)
	{
		c1 = 1.0;
		c2 = 2.0;
		c3 = 1.0;
	}
	else //use Scharr mask, expermental
	{
		c1 = 3.0;
		c2 = 10.0;
		c3 = 3.0;
	}
	
	fraction = (1.0/(c1+c2+c3));

	cv::Mat2d indices(m_msi.height, m_msi.width);
	for (int y = 0; y < m_msi.height ; y++)
	{
		cv::Vec2d *drow = indices[y];
		for (int x = 0; x < m_msi.width ; x++)
		{
			const cv::Point &p = m_som.identifyWinnerNeuron(y,x);
			drow[x][0] = p.x;
			drow[x][1] = p.y;
		}
	}

	double valx,valy;
	bool periodic;
	if(config.graph_type == "MESH_P")
		periodic = true;
	else
		periodic = false;	
	
	double maxIntensity = 0.0;
	
	unsigned int ten = (m_msi.height* m_msi.width)/10;
	int round = 1;
	
	if(config.verbosity > 0)
		std::cout << "  0 %" <<std::endl;
	for (int y = 1; y < m_msi.height-1; y++)
	{
		double* x_ptr = dx[y];
		double* y_ptr = dy[y];
		cv::Vec2d *i_ptr = indices[y];
		cv::Vec2d *i_uptr = indices[y-1];
		cv::Vec2d *i_dptr = indices[y+1];
		double xx, yy;
	
		for (int x = 1; x < m_msi.width-1; x++)
		{
			if(( (y*m_msi.width + x )% ten) == 0 && config.verbosity > 0)
			{	
				std::cout << " " << round * 10 << " %" <<std::endl;
				round++;
			}	
			{	// y-direction
				cv::Point2d u,d;

				xx = (c1 * i_uptr[x-1][0] + c2 * i_uptr[x][0] + c3 * i_uptr[x+1][0]) * fraction;
				yy = (c1 * i_uptr[x-1][1] + c2 * i_uptr[x][1] + c3 * i_uptr[x+1][1]) * fraction;

				u.x = xx;
				u.y = yy;
				xx = (c1 * i_dptr[x-1][0] + c2 * i_dptr[x][0] + c3 * i_dptr[x+1][0]) * fraction;
				yy = (c1 * i_dptr[x-1][1] + c2 * i_dptr[x][1] + c3 * i_dptr[x+1][1]) * fraction;

				d.x = xx;
				d.y = yy;

				if(m_withGraph || m_withUMap)
					valy = graphDistance(u, d);
				else
					valy = vectorDistance(u, d);

				if (maxIntensity < valy)
					maxIntensity = valy;
				if ((u.x*u.x + u.y*u.y) < (d.x*d.x + d.y*d.y) )
					valy = -valy;
				y_ptr[x] =  valy;
			}
			{	// x-direction
				cv::Point2d u,d;

				xx = (c1 * i_uptr[x-1][0] + c2 * i_ptr[x-1][0] + c3 * i_dptr[x-1][0]) * fraction;
				yy = (c1 * i_uptr[x-1][1] + c2 * i_ptr[x-1][1] + c3 * i_dptr[x-1][1]) * fraction;

				u.x = xx;
				u.y = yy;
				xx = (c1 * i_uptr[x+1][0] + c2 * i_ptr[x+1][0] + c3 * i_dptr[x+1][0]) * fraction;
				yy = (c1 * i_uptr[x+1][1] + c2 * i_ptr[x+1][1] + c3 * i_dptr[x+1][1]) * fraction;

				d.x = xx;
				d.y = yy;
				
				if(m_withGraph || m_withUMap)
					valx = graphDistance(u, d);
				else
					valx = vectorDistance(u, d);
								
				if (maxIntensity < valx)
					maxIntensity = valx;

				if ((u.x*u.x + u.y*u.y) < (d.x*d.x + d.y*d.y) )
					valx = -valx;
				x_ptr[x] = valx;
			}
		}
	}
  //normalization
	for (int y = 0; y < m_msi.height ; y++)
	{
		double* x_ptr = dx[y];
		double* y_ptr = dy[y];

		for(int x = 0; x < m_msi.width ; x++)
		{
			x_ptr[x] = ( ((x_ptr[x] + maxIntensity)*0.5/maxIntensity));
			y_ptr[x] =  ( ((y_ptr[x] + maxIntensity)*0.5/maxIntensity));
		}
  }
  if(config.verbosity > 0)
		std::cout << "100 %" <<std::endl;
}

double SomTrainer::vectorDistance(const cv::Point2d &p1, const cv::Point2d &p2)
{
	double dx, dy;

	dx = (p1.x - p2.x); dy = (p1.y - p2.y);
	return std::sqrt(dx * dx + dy * dy);
}

double SomTrainer::wrapAroundDistance( cv::Point2d &p1, cv::Point2d &p2)
{
	cv::Point2d direction = p2 - p1;
	cv::Point2d intersection;
	cv::Point2d inter1, inter2;
		
	int bordersFound = 0;
	cv::Mat solution;
	double h = (double)(m_heightSom);
	double w = (double)(m_widthSom);
	
	cv::Mat1d lefthandside = cv::Mat::zeros(2,2,CV_64F);
	cv::Mat1d righthandside = cv::Mat::zeros(2,1,CV_64F); 
	
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
		return vectorDistance(p1,p2);
	else
	{
		if(config.sw_initialDegree == 4)
		{
			return (manhattanDistance(inter1,inter2) - manhattanDistance(p1,p2));
		}	
		else
			return (euclideanDistance(inter1,inter2) - euclideanDistance(p1,p2));
	}
}

bool SomTrainer::findBorderIntersection(cv::Point2d &p1, cv::Point2d &p2, cv::Point2d &intersect,cv::Mat1d &border)
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

double SomTrainer::graphDistance( cv::Point2d &p1, cv::Point2d &p2)
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

	index1 = (unsigned int)cp1.y * m_som.getWidth() + (unsigned int)cp1.x;
	index2 = (unsigned int)cp2.y * m_som.getWidth() + (unsigned int)cp2.x;

	distance = 0.0;
	
	if(config.graph_withGraph == false && config.withUMap == true)
	{
		assert(!umap.empty());

		distance = msi_graph->getDistance(index1,index2,true);
	}	
		
	else if(config.graph_withGraph == true && config.withUMap == true)
	{	
		distance = msi_graph->getDistance(index1,index2, true);
	}	

	else if(config.graph_withGraph == true && config.withUMap == false)
	{	
		if(periodic)
		{
			if(config.forceDD)
			{
				double dd,wd;
				dd = vectorDistance(cp1,cp2);
				wd = wrapAroundDistance(cp1,cp2);

				distance = std::min(dd, wd);
			}	
			else
				distance = msi_graph->getDistance(index1,index2,false);
		}
		else
			distance = msi_graph->getDistance(index1,index2,false);
	}	

	//actually, should never eb reached
	if(distance == DBL_MAX)
	{
		std::cout <<" infinite distance!! " << index1 << " " << index2 << " " <<distance<<std::endl;
		return 1.0;
	}	
	
	return distance;
}

cv::Mat SomTrainer::generateEdgeImage(double h1, double h2) 
{

	m_edge = cv::Mat_<uchar>(m_img_rank.size());
	cv::Mat_<uchar> edgeShow;

	for(int y = 0; y < m_img_rank.rows; y++) 
	{
		for(int x = 0; x < m_img_rank.cols; x++) 
		{
			m_edge.at<uchar>(y,x) = static_cast<uchar>(m_img_rank.at<float>(y,x) *255.0f );
		}
	}

	cv::Canny( m_edge, edgeShow, h1, h2, 3, true );

	cv::imwrite(config.output_dir+getFilenameExtension()+"_edge.png", edgeShow);

	return m_edge;
}

void SomTrainer::processStatistics() 
{

	m_evolution = m_evolution / (double)m_iter;

	m_foundEdges = 1;
	int increasing = 0;
	int decreasing = 0;
	unsigned short color = m_img_rank.at<ushort>(0,0);
	for(int x = 1; x < m_msi.width; x++) 
	{
		if(color < m_img_rank.at<ushort>(0,x)) increasing++;
		if(color > m_img_rank.at<ushort>(0,x)) decreasing++;
		color = m_img_rank.at<ushort>(0,x);
	}
	m_foundEdges = std::max(decreasing, increasing);

  // open stream to write at
	m_balance = generateBWSom();

	if(m_verbosity > 1) 
	{
		std::ofstream params( (config.output_dir+m_msiName + "_params.txt").c_str(), std::ofstream::app | std::ofstream::out );
		if(params.bad())
			std::cerr << "Params writing not possible!" << std::endl;
		params << m_widthSom << " " << m_heightSom << " " << m_msi.size() << " " << m_radiusStart << " " << m_radiusEnd;
		params << " " << m_iter << " " << m_learningStart << " " << m_learningEnd << " " << m_quantizationError << " " <<  m_errorEntropy << " " << m_evolution << " " << m_balance << " " << m_foundEdges << std::endl;
		params.close();
	}
}

std::string SomTrainer::getParams(bool measurements) 
{
	std::stringstream params;
	params << m_widthSom << " " << m_heightSom << " " << m_bands << " " << m_radiusStart << " " << m_radiusEnd;
	params << " " << m_iter << " " << m_learningStart << " " << m_learningEnd;
	if(measurements) 
		params << " " << m_evolution;

	return params.str();
}

std::vector<double> SomTrainer::getStatistics() 
{

	std::vector<double> stats;
	stats.push_back(m_widthSom); stats.push_back(m_heightSom); stats.push_back(m_bands);
	stats.push_back(m_radiusStart); stats.push_back(m_radiusEnd); stats.push_back(m_iter); 
	stats.push_back(m_learningStart); stats.push_back(m_learningEnd); stats.push_back(m_quantizationError);
	stats.push_back(m_errorEntropy); stats.push_back(m_evolution); stats.push_back(m_balance); stats.push_back(m_foundEdges);

	return stats;
}

std::string SomTrainer::getFilenameExtension() 
{

	std::stringstream s1, s2, s3, s4, s5, s6, s7, s8;
	std::string height, width, iter, radiusS, radiusE, learnS, learnE;
	std::string graphDistance,neighbors,  sw_type , sw_value;

	s1 << m_heightSom;
	height = s1.str();
	s2 << m_widthSom;
	width = s2.str();
	s3 << m_iter;
	iter = s3.str();
	s4 << m_radiusStart;
	radiusS = s4.str();
	s5 << m_radiusEnd;
	radiusE = s5.str();
	s6 << m_learningStart;
	learnS = s6.str();
	s7 << m_learningEnd;
	learnE = s7.str();
	// SOM is computed the basic way
	if(config.graph_withGraph == false)
	{	
		s8 << "_DDIST";
		//Edge detection computes via umap (graph based or again direct??)
		if(config.withUMap)
			s8 << "_UMAP_" << config.scaleUDistance ;
	}	
	//SOM is trained on a graph topology
	else
	{	
		s8 << "_GDIST";
		if(config.withUMap)
			s8 << "_UMAP_"<< config.scaleUDistance;
		//with N neighboring nodes
		s8 << "_N";
		// using wrap around (periodic) structure
		if(config.graph_type == "MESH_P")
			s8 << "P";	
		s8 << config.sw_initialDegree;
		if(config.forceDD)
			s8 << "_FDD";
		// graph is a small world graph using phi-model
		// phi percent of edges are SHORTCUTS
		if(config.sw_model == "PHI" && (config.sw_phi != 0.0 ))
		{
			s8 << "_PHI";
			s8 << config.sw_phi;
		}
		// graph is a small world graph using beta-model
		// beta percent of edges are rewired
		else if(config.sw_model == "BETA" && (config.sw_beta != 0.0 ))
		{
			s8 << "_BETA";
			s8 << config.sw_beta;		
		}
		// graph is basic N(P)-connected mesh
		else
		{
			//distances are computed via umap
			if(config.withUMap)
				s8 << "_UMAP";
		}	
	}	
  
	graphDistance = s8.str();
  
	std::string name(m_msiName);

	return name + "_"+height+"x"+width+"_"+iter+ "_R"+radiusS+"-"+radiusE+ "_L" + learnS +"-"+ learnE +graphDistance;
}
