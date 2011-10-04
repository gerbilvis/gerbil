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


#include <iostream> 
#include <cassert> 
#include <string>
#include <vector>

#include "cv.h"
#include "gmm.h"

#include "stopwatch.h"

GMM::GMM(unsigned int dim, unsigned int ncentres, std::string covar_type, S_GMMNet &net, bool fixedSeed)
{
	assert(ncentres > 0);
	assert( covar_type == "SPHERICAL" || covar_type == "DIAG" || covar_type == "FULL" || covar_type == "PPCA");
  
	cv::RNG rng;
	uint64 state = 19;
	
	if(fixedSeed)
		rng = cv::RNG(cv::getTickCount());
	else
		rng = cv::RNG(state);
  
  net.type = "gmm";
  net.nin = dim;
  net.ncentres = ncentres;
  
  //only this mode is implemented
  assert(covar_type == "SPHERICAL");
  
  net.covar_type = covar_type;
  net.priors = cv::Mat::ones(1, net.ncentres, CV_64F);
  net.priors.setTo(1.0/(double)net.ncentres);
  
  net.centres = cv::Mat::zeros(net.ncentres, net.nin, CV_64F);
// 	cv::Mat1d w;
// 	const char *fn  = "/home/knorx/UNI/Studienarbeit/working_directory/code/reflectance/ms_edges/GTM/data/centers.dat";
// 	loadFromFile(net.centres, fn);
  rng.fill(net.centres, cv::RNG::NORMAL, cv::Scalar(0.0),cv::Scalar(1.0));
  if(covar_type == "SPHERICAL")
  {
    net.covars = cv::Mat::ones(1, net.ncentres, CV_64F);
    net.nwts = net.ncentres + (net.ncentres * net.nin) + net.ncentres;
  }
  else
    std::cerr << "Not implemented" <<std::endl;
	
}

void GMM::gmmpost(S_GMMNet &mix,cv::Mat1d &x,cv::Mat1d &post,cv::Mat1d &activations)
{
	unsigned int ndata = x.rows;

	gmmactiv(mix, x , activations);
	
	post = cv::Mat::zeros(ndata, mix.ncentres,CV_64F);
	cv::Mat1d post_ones = cv::Mat::ones(ndata, 1,CV_64F);

	cv::multiply( (post_ones * mix.priors), activations, post );
	
	cv::Mat1d s = cv::Mat::zeros(ndata,1,CV_64F);
	
	std::vector<unsigned int> zero_rows;
	
	//calculate the margin
	for(unsigned int y = 0; y < ndata; y++ )
	{
		double *rowPtr = s[y];
		double *postPtr = post[y];
		for(unsigned int x = 0; x < mix.ncentres; x++ )
		{
			rowPtr[0] += postPtr[x];
		}
		
		//check for zero probability
		if(rowPtr[0] == 0.0)
		{	
			rowPtr[0] = 1.0;
			zero_rows.push_back(y);
		}	
	}
	//check for zero probability
	if(zero_rows.size() > 0)
	{

		for(unsigned int i = 0; i < zero_rows.size();i++)
		{
			//fetch index of zero row
			double *rowPtr = post[zero_rows.at(i)];
			for(unsigned int x = 0; x < mix.ncentres;x++)
			{
				//set any zeros to 1 before dividing
				rowPtr[x] = 1.0/(double)mix.ncentres;
			}	
		}	
	}
	
	cv::Mat1d tmp = cv::Mat1d(ndata, mix.ncentres,CV_64F);
	cv::Mat1d res;
	tmp = cv::Mat::ones(1, mix.ncentres,CV_64F);
	cv::divide(post, (s * tmp), res);
	
	post = res;
		
}

void GMM::gmmactiv(S_GMMNet &mix,cv::Mat1d &x ,cv::Mat1d &a)
{
	//only spherical implemented here
	assert(mix.covar_type == "SPHERICAL");

	unsigned int ndata = x.rows;
	
	a = cv::Mat::zeros(ndata, mix.ncentres,CV_64F);
	
	if(mix.covar_type == "SPHERICAL")
		
	{
		cv::Mat1d distances;
		
		//calculate squared norm matrix
		dist2(x , mix.centres, distances);

		//calculate width factors
		cv::Mat1d wi2 = cv::Mat::zeros(ndata, mix.ncentres, CV_64F);
		cv::Mat1d tmp = cv::Mat::ones(ndata,1,CV_64F);
		
		wi2 = ( tmp * (2.0 * mix.covars) );
		cv::Mat1d normal;
		cv::pow( (CV_PI * wi2),((double)mix.nin/2.0) ,normal );
	
		//now compute the activations
		for(unsigned int y = 0; y < ndata; y++)
		{
			double *rowPtr = a[y];
			for(unsigned int x = 0; x < mix.ncentres; x++)
			{
				double val = std::exp(- (distances[y][x]/wi2[y][x])) / normal[y][x];
				//without this check, NAN vlues can destroy the whole system
				if(val < std::exp(-30.0))val = 0.0;
				rowPtr[x] = val;
				
			}
		}
			
	}
	else if(mix.covar_type == "DIAG")
	{
		//TODO not implemented here
	}
	else if(mix.covar_type == "FULL")
	{
		//TODO not implemented here
	}
	else if(mix.covar_type == "PPCA")
	{
		//TODO not implemented here
	}
		
}

void GMM::gmmprob(S_GMMNet &mix,cv::Mat1d &data, cv::Mat1d &prob)
{
	cv::Mat1d activations, priors_trans;
	prob = cv::Mat::zeros(data.rows, 1U, CV_64F);
	
	//compute activations
	gmmactiv(mix,data, activations);
	
	cv::transpose(mix.priors, priors_trans);
	prob = activations * priors_trans;
	
}

void GMM::dumpNet(S_GMMNet &net)
{
  std::cout << "+++ Content of NET structure +++\n" <<std::endl;
  std::cout << "Type: " <<net.type <<std::endl;
  std::cout << "nin: " <<net.nin <<std::endl;
  std::cout << "ncentres: " <<net.ncentres <<std::endl;
  std::cout << "covar_type: " <<net.covar_type <<std::endl;
  
  
  std::cout << "\nmatric priors: " << "1 x " << net.ncentres << "\n"<<std::endl;
	std::cout << "\nmatric priors: "<< net.priors << std::endl;

  
  std::cout << "\nmatric centres: " << net.ncentres<<" x " << net.nin << "\n"<<std::endl;
	std::cout << "\nmatric centres: "<< net.centres << std::endl;
  
  std::cout << "\nmatric covars: " << "1 x " << net.ncentres<< "\n" <<std::endl;
	std::cout << "\nmatric covars: "<< net.covars << std::endl;

}

GMM::~GMM()
{}
 
