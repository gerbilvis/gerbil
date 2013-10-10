/*
	Copyright(c) 2012 Ralph Muessig	and Johannes Jordan
	<johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "som_trainer.h"

#include <progress_observer.h>

#include <stopwatch.h>
#include <opencv2/core/core.hpp>

#include <iostream>
#include <fstream>

SOMTrainer::SOMTrainer(SOM *map, const multi_img &image,
						 const vole::EdgeDetectionConfig &conf)
	: som(map), input(image), config(conf), currIter(0),
		   m_bmuMap(cv::Mat::zeros(som->get2dHeight(), som->get2dWidth(), CV_64F)),
		   abort(NULL), po(NULL)
{}

SOMTrainer::SOMTrainer(SOM *map, const multi_img &image,
						 const vole::EdgeDetectionConfig &conf, volatile bool &abort,
						vole::ProgressObserver *po)
	: som(map), input(image), config(conf), currIter(0),
	  m_bmuMap(cv::Mat::zeros(som->get2dHeight(), som->get2dWidth(), CV_64F)),
	  abort(&abort), po(po)
{
}

SOM *SOMTrainer::train(const vole::EdgeDetectionConfig &conf, const multi_img &img)
{
	bool dummy;
	return train(conf, img, dummy, NULL);
}

SOM *SOMTrainer::train(const vole::EdgeDetectionConfig &config,
						 const multi_img &img, volatile bool &abort,
					   vole::ProgressObserver *po
					   )
{
	if (config.som_file.empty()) {
		vole::Stopwatch running_time("Total running time");
		SOM *som = SOM::createSOM(config, img.size(), img.meta);
		std::cout << "# Generated " << som->description() << std::endl;

		SOMTrainer trainer(som, img, config, abort, po);

		std::cout << "# SOM Trainer starts to feed the network using "
				  << config.maxIter << " iterations..." << std::endl;

		vole::Stopwatch watch("Training");
		trainer.feedNetwork();
		return som;
	} else {
		// no progress info for po, probably doesn't hurt (?)
		multi_img somimg;
		somimg.minval = img.minval;
		somimg.maxval = img.maxval;
		somimg.read_image(config.som_file);
		if (somimg.empty()) {
			std::cerr << "Could not read image containing the SOM!" << std::endl;
			return NULL;
		}
		if (somimg.size() != img.size()) {
			std::cerr << "Somimg and input have a different amount of bands!" << std::endl;
			return NULL;
		}
		somimg.rebuildPixels(false);
		return SOM::createSOM(config, somimg, img.meta);
	}
}

void SOMTrainer::feedNetwork()
{
	// matrices that hold shuffled sequences of the input for number of iterations
	std::cout << "Start feeding" << std::endl;
	cv::Mat_<int> shuffledY(1, config.maxIter);
	cv::Mat_<int> shuffledX(1, config.maxIter);

	cv::RNG rng(config.seed);

	// generate random sequence of the input x,y range
	rng.fill(shuffledY, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(input.height));
	rng.fill(shuffledX, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(input.width));

	// start the iterative feeding process here
	cv::MatConstIterator_<int> itY = shuffledY.begin();
	cv::MatConstIterator_<int> itX = shuffledX.begin();

	// output percentage
	unsigned int hundred = std::max<unsigned int>(config.maxIter/100, 100);
	int round = 1;
	if (config.verbosity > 0)
		std::cout  << "  0 %"; std::cout.flush();
	long sumOfUpdates = 0;
	int ctr = 0;
	for (; itX != shuffledX.end(); itX++, itY++)
	{
		if(abort != NULL && *abort == true) {
			std::cerr << "SOMTRainer aborting" << std::endl;
			break;
		}
		// extract random pixel vector from the multispectral image
		const multi_img::Pixel &vec = input(*itY, *itX);
		sumOfUpdates += feedSample(vec);
		if ((config.verbosity > 0) && (config.maxIter > 100)
			&& ((currIter % hundred) == 0) && (round < 100)) {
			std::cout << "\r" << (round < 10 ? "  " : " ") << round << " %";
			std::cout.flush();
			round++;
		}
		++ctr;
		// print update statistic each 1%
		if (ctr % (config.maxIter / 100) == 0)
		{
			// send progress updates if we have an observer
			po && po->update(int(ctr/float(config.maxIter)*100));
			if(config.verbosity >= 2) {
				std::cout << " Feed #" << ctr
						  << ", mean of neuron-updates in the last "
						  << (config.maxIter / 100) << " iterations: "
						  << ((double)sumOfUpdates / (config.maxIter / 100))
						  << std::endl;
				sumOfUpdates = 0;
			}
		}
		// print som each 20%
		if (config.verbosity >= 3 && ctr % (config.maxIter / 5) == 0)
		{
			multi_img somimg = som->export_2d();
			std::ostringstream filename;
			filename << "debug_som_" << (ctr / (config.maxIter / 5));
			somimg.write_out(filename.str());
		}
	}
	if(config.verbosity > 0)
		std::cout << "\r100 %" <<std::endl;

	std::cout <<"# Feeding done" <<std::endl;

	// write the visualization of SOM
	//if(config.verbosity > 2 ) {
	//	displaySom(true);
	//	displayMSI(false);
	//	displayBmuMap(false);

	//	if (config.withUMap)
	//		umatrix(false);
	//	if(config.isGraphical && m_withGraph)
	//	{
	//	compareMultispectralData();
	//		displayGraphDistances();
	//	}
	//}
}

int SOMTrainer::feedSample(const multi_img::Pixel &input)
{
	// adjust learning rate and radius
	// note that they are _decreasing_ -> start * (end/start)^(iter%)
	double learnRate = config.learnStart * std::pow(
				config.learnEnd / config.learnStart,
				(double)currIter/(double)config.maxIter);
	double sigma = config.sigmaStart * std::pow(
				config.sigmaEnd / config.sigmaStart,
				(double)currIter/(double)config.maxIter);

	// find best matching neuron to given input vector
	SOM::iterator neuron = som->identifyWinnerNeuron(input);
	cv::Point pos = neuron.get2dCoordinates();

	// increase winning count of neuron
	m_bmuMap(pos) += 1.0;
	// BMU and their neighborhood learns weighted from the input | wtf?!

	//vole::Stopwatch watch;
	int updates = som->updateNeighborhood(neuron, input, sigma, learnRate);
	//watch.print("Neighborhood updated");

	currIter++;
	return updates;
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
					cv::PointtopCenter = som.getNeuron(x,h);
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

	int width = som->get2dWidth();
	int height = som->get2dHeight();

	m_img_bwsom = cv::Mat_<float>(height, width);

	int radius = 1;

	// determine radius
	// TODO: what is done here and why? Important: we use the squared distance
	//       in the loop and should have a squared radius!
	if (height % 2 == 0) {
		int order = 0;
		int length = 2;
		for(order = 1; order < 10; order++) {
			length *= 2;
			if (length == height) { break; } // TODO: andere Reihenfolge absicht?
		}
		radius = order;//static_cast<unsigned int>(pow(2., order));
	} else if (height % 3 == 0) {
		int order = 0;
		int length = 3;
		for(order = 1; order < 10; order++) {
			if (length == height) { break; } // TODO: andere Reihenfolge absicht?
			length *= 3;
		}
		radius = order;//static_cast<unsigned int>(pow(2., order));
	}
	if (height == 1) {
		if (width < 16) {
			radius = 3;
		} else {
			radius = 1 + width/10;
		}
	}
	radius = 1;

	double totalDiff = 0.;

	const SOM::iterator theEnd = som->end();
	for (SOM::iterator n = som->begin(); n != theEnd; ++n) {
		double difference = 0.;
		int count = 0;

		for (SOM::neighbourIterator neighbour = n.neighboursBegin(radius);
			neighbour != n.neighboursEnd(radius); ++neighbour) {
			count++;
			difference += som->getSimilarity(*n, *neighbour);
		}

		if(count > 1) { difference = difference / static_cast<double>(count-1); }
		totalDiff += difference;
		cv::Point pos = n.get2dCoordinates();
		m_img_bwsom(pos) = static_cast<float>(difference);
	}
	// normalize on number of neurons
	totalDiff /= (double)som->size();

	return totalDiff;
}
