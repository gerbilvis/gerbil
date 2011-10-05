#include "som_trainer.h"

#include <stopwatch.h>

#include <cv.h>

#include <iostream>
#include <fstream>

SOMTrainer::SOMTrainer(SOM &map, const multi_img &image,
                     const EdgeDetectionConfig &conf)
  : som(map), input(image), config(conf)
{

  m_nr = 0;
  currIter = 0;
  
  m_bmuMap = cv::Mat::zeros(m_heightSom, m_widthSom, CV_64F);
  
  m_withUMap = conf.withUMap;
  m_withGraph = som.withGraph();

	if (m_withGraph || m_withUMap)
  {
    som.getGraph()->precomputePaths(false);
  }
}

void SOMTrainer::feedNetwork()
{

  // matrices that hold shuffled sequences of the input for number of iterations
	std::cout << "# Start feeding"  <<std::endl;
	cv::Mat_<int> shuffledY(1, maxIter);
	cv::Mat_<int> shuffledX(1, maxIter);
  
	cv::RNG rng;
	
	if(config.fixedSeed)
		rng = cv::RNG (19.0); // TODO
	else
		rng = cv::RNG(cv::getTickCount());
	
  // generate random sequence of the input x,y range
	rng.fill( shuffledY, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(input.height));
	rng.fill( shuffledX, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(input.width));
  
  // start the iterative feeding process here
	cv::MatConstIterator_<int> itY = shuffledY.begin();
	cv::MatConstIterator_<int> itX = shuffledX.begin();

	// output percentage	
	unsigned int ten = std::max<unsigned int>(maxIter/10, 10);
	int round = 1;
	if(config.verbosity > 0)
		std::cout  << "  0 %" <<std::endl;
	for (; itX != shuffledX.end(); itX++, itY++) 
	{
		// extract random pixel vector from the multispectral image
		const multi_img::Pixel & vec = input(*itY, *itX);
		feedSample(vec);
		if( (maxIter > 10) && (currIter % ten) == 0 && config.verbosity > 0) {
			std::cout << " "<< (round*10) << " %" <<std::endl;
			round++;
		}	
  }
  if(config.verbosity > 0)
		std::cout  << "100 %" <<std::endl;

  std::cout <<"# Feeding done" <<std::endl;

	// write the visualization of SOM
	if(config.verbosity > 2 ) {
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
}

void SOMTrainer::feedSample(const multi_img::Pixel &input)
{
	// adjust learning rate and radius
	double learnRate = config.som_learnStart * std::pow(
	            config.som_learnStart / config.som_learnEnd,
	            (double)currIter/(double)maxIter);
	double radius = config.som_radiusStart * std::pow(
	            config.som_radiusStart / config.som_radiusEnd,
	            (double)currIter/(double)maxIter);

	// find best matching neuron to given input vector
	cv::Point pos = som.identifyWinnerNeuron(input);

	//increase winning count of neuron
	m_bmuMap[pos.y][pos.x] += 1.0;
	// BMU and their neigborhood learns weighted from the input

	//vole::Stopwatch watch;

	updateNeighborhood(pos, input, radius, learnRate);
	//watch.print("Neighborhood updated");

	currIter++;
}

cv::Mat1d SOMTrainer::umatrix(bool write)
{
	int width = som.getWidth();
	int height = som.getHeight();
		
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
	int neighbors;

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
				center = som.getNeuron(x,y);
				
			if(t == false)
			{	
				topCenter = som.getNeuron(x,y-1);
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
					topCenter = som.getNeuron(x,h);
					multi_img::Pixel p_tc(*topCenter);
					cross += center->euclideanDistance(p_tc);
					msi_graph->setEdgeWeight((y*width +x),((h)*width +x), center->euclideanDistance(p_tc));
					neighbors++;
				}	
			}	
				
			if(l == false)
			{	
				centerLeft = som.getNeuron(x-1,y);
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
					centerLeft = som.getNeuron(w,y);
					multi_img::Pixel p_cl(*centerLeft);
					cross += center->euclideanDistance(p_cl);
					msi_graph->setEdgeWeight((y*width +x),((y)*width +w), center->euclideanDistance(p_cl));
					neighbors++;
				}	
			}	
				
			if(r == false)
			{	
				centerRight = som.getNeuron(x+1,y);
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
					centerRight = som.getNeuron(w,y);
					multi_img::Pixel p_cr(*centerRight);
					cross += center->euclideanDistance(p_cr);
					msi_graph->setEdgeWeight((y*width +x),((y)*width +w), center->euclideanDistance(p_cr));
					neighbors++;
				}	
			}	
				
			if(b == false)
			{	
				bottomCenter = som.getNeuron(x,y+1);
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
					bottomCenter = som.getNeuron(x,h);
					multi_img::Pixel p_bc(*bottomCenter);
					cross += center->euclideanDistance(p_bc);
					msi_graph->setEdgeWeight((y*width +x),((h)*width +x), center->euclideanDistance(p_bc));
					neighbors++;
				}	
			}	
//diagonal edges
			if((l | t) == false && (initialDegree == 8))
			{
				topLeft = som.getNeuron(x-1,y-1);
				multi_img::Pixel p_tl(*topLeft);

				diagonal += center->euclideanDistance(p_tl) * (1.0/std::sqrt(2.0));
				msi_graph->setEdgeWeight((y*width +x),((y-1)*width +(x-1)), center->euclideanDistance(p_tl) );
				neighbors++;
			}
			
			if((r | t) == false && (initialDegree == 8)	)
			{	
				topRight = som.getNeuron(x+1,y-1);
				multi_img::Pixel p_tr(*topRight);

				diagonal += center->euclideanDistance(p_tr) * (1.0/std::sqrt(2.0));
				msi_graph->setEdgeWeight((y*width +x),((y-1)*width +(x+1)), center->euclideanDistance(p_tr) );
				neighbors++;
			}
					
			if((b | l) == false && (initialDegree == 8))
			{	
				bottomLeft = som.getNeuron(x-1,y+1);
				multi_img::Pixel p_bl(*bottomLeft);

				diagonal += center->euclideanDistance(p_bl) * (1.0/std::sqrt(2.0));	
				msi_graph->setEdgeWeight((y*width +x),((y+1)*width +(x-1)), center->euclideanDistance(p_bl));
				neighbors++;
			}
				
			if((b | r) == false && (initialDegree == 8))
			{	
				bottomRight = som.getNeuron(x+1,y+1);
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


double SOMTrainer::generateBWSom() {

  int width = som.getWidth();
  int height = som.getHeight();

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
			const Neuron *cn = som.getNeuron(x,y);
      int count = 0;

      for(int posY = y-radius; posY <= y+radius; posY++) {
        if(posY < 0 || posY >= height) { continue; }
        for(int posX = x-radius; posX <= x+radius; posX++) {
          if(posX < 0 || posX >= width) { continue; }
          //double dist = (x - posX)*(x - posX) + (y - posY)*(y - posY);
        //  if(dist <= radius) {
            count++;
					const Neuron *neighbor = som.getNeuron(posX,posY);
					difference += config.distfun->getSimilarity(*cn, *neighbor);
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

double SOMTrainer::vectorDistance(const cv::Point2d &p1, const cv::Point2d &p2)
{
	double dx, dy;

	dx = (p1.x - p2.x); dy = (p1.y - p2.y);
	return std::sqrt(dx * dx + dy * dy);
}

double SOMTrainer::wrapAroundDistance( cv::Point2d &p1, cv::Point2d &p2)
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

bool SOMTrainer::findBorderIntersection(cv::Point2d &p1, cv::Point2d &p2, cv::Point2d &intersect,cv::Mat1d &border)
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

double SOMTrainer::graphDistance(cv::Point2d &p1, cv::Point2d &p2)
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

	index1 = (unsigned int)cp1.y * som.getWidth() + (unsigned int)cp1.x;
	index2 = (unsigned int)cp2.y * som.getWidth() + (unsigned int)cp2.x;

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

	//actually, should never be reached
	if(distance == DBL_MAX)
	{
		std::cout <<" infinite distance!! " << index1 << " " << index2 << " " <<distance<<std::endl;
		return 1.0;
	}	
	
	return distance;
}
