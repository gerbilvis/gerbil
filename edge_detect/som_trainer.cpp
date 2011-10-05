#include "som_trainer.h"

#include <stopwatch.h>

#include <cv.h>

#include <iostream>
#include <fstream>

SOMTrainer::SOMTrainer(SOM &map, const multi_img &image,
                     const EdgeDetectionConfig &conf)
    : som(map), input(image), config(conf), currIter(0)
{
  m_bmuMap = cv::Mat::zeros(som.getHeight(), som.getWidth(), CV_64F);
}

void SOMTrainer::feedNetwork()
{

  // matrices that hold shuffled sequences of the input for number of iterations
	std::cout << "Start feeding"  <<std::endl;
	cv::Mat_<int> shuffledY(1, maxIter);
	cv::Mat_<int> shuffledX(1, maxIter);
  
	cv::RNG rng;
	
	if(config.fixedSeed)
		rng = cv::RNG (19.0); // TODO
	else
		rng = cv::RNG(cv::getTickCount());
	
  // generate random sequence of the input x,y range
	rng.fill(shuffledY, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(input.height));
	rng.fill(shuffledX, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(input.width));
  
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
	//	displaySom(true);
	//	displayMSI(false);
	//	displayBmuMap(false);

//		if (config.withUMap)
//			umatrix(false);	 // TODO: do we need this for operation? if yes, bug!
	//	if(config.isGraphical && m_withGraph)
	//	{
	//	compareMultispectralData();
	//		displayGraphDistances();
	//	}
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
	m_bmuMap(pos) += 1.0;
	// BMU and their neighborhood learns weighted from the input | wtf?!

	//vole::Stopwatch watch;
	som.updateNeighborhood(pos, input, radius, learnRate);
	//watch.print("Neighborhood updated");

	currIter++;
}

/*
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
		
	if (config.withUMap)
		som.getGraph ... msi_graph->scaleDistances(config.scaleUDistance);

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
		
		std::string graph = "/";
		if (m_withGraph) {
			graph += "graph";
			if(config.sw_phi > 0.0 || config.sw_beta > 0.0)
				graph += "sw_graph";
		}	
		cv::imwrite(config.output_dir + graph + "_umatrix.png", umatrix);
	
		//write gnuplot data
		std::ofstream out;
		std::string fn = config.output_dir + graph +"_umatrix.dat";
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
*/

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
